# -*- coding: utf-8 -*-
import argparse
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from model import build_model, load_weights


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def batched(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def maybe_resize(image: Image.Image, long_edge: int | None) -> Tuple[Image.Image, Tuple[int, int]]:
    original_size = image.size
    if long_edge is None or long_edge <= 0:
        return image, original_size
    width, height = image.size
    current_long = max(width, height)
    if current_long <= long_edge:
        return image, original_size
    scale = long_edge / current_long
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS), original_size


def preprocess(path: Path, resize_long_edge: int | None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    image, original_size = maybe_resize(image, resize_long_edge)
    tensor = TF.normalize(TF.to_tensor(image), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return tensor, original_size


def pad_batch_to_common_size(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    max_h = max(height for height, _ in sizes)
    max_w = max(width for _, width in sizes)
    padded = []
    for tensor in tensors:
        height, width = tensor.shape[-2:]
        padded.append(F.pad(tensor, (0, max_w - width, 0, max_h - height), mode="constant", value=0.0))
    return torch.stack(padded, dim=0), sizes


def save_output(tensor: torch.Tensor, out_path: Path, model_size: Tuple[int, int], original_size: Tuple[int, int]) -> None:
    height, width = model_size
    image = TF.to_pil_image(tensor[:, :height, :width].clamp(0.0, 1.0).cpu())
    if image.size != original_size:
        image = image.resize(original_size, Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def tta_transforms(mode: str):
    if mode == "none":
        return [(lambda x: x, lambda y: y)]
    x4_transforms = [
        (lambda x: x, lambda y: y),
        (lambda x: torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),
        (lambda x: torch.flip(x, dims=[-2]), lambda y: torch.flip(y, dims=[-2])),
        (lambda x: torch.flip(x, dims=[-2, -1]), lambda y: torch.flip(y, dims=[-2, -1])),
    ]
    if mode == "x4":
        return x4_transforms
    if mode == "d4":
        transforms = []
        for rotation in range(4):
            for hflip in (False, True):
                def augment(x, rotation=rotation, hflip=hflip):
                    y = torch.rot90(x, k=rotation, dims=(-2, -1))
                    if hflip:
                        y = torch.flip(y, dims=[-1])
                    return y

                def deaugment(y, rotation=rotation, hflip=hflip):
                    if hflip:
                        y = torch.flip(y, dims=[-1])
                    return torch.rot90(y, k=(-rotation) % 4, dims=(-2, -1))

                transforms.append((augment, deaugment))
        return transforms
    raise ValueError(f"unknown TTA mode: {mode}")


@torch.inference_mode()
def run_model(model, batch: torch.Tensor, args, device: torch.device) -> torch.Tensor:
    outputs = []
    for augment, deaugment in tta_transforms(args.tta):
        aug_batch = augment(batch)
        if args.amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(aug_batch)
        else:
            pred = model(aug_batch)
        outputs.append(deaugment(pred).float())
    return torch.stack(outputs, dim=0).mean(dim=0)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args():
    parser = argparse.ArgumentParser("Stage1 folder inference")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", "--bs", type=int, default=1)
    parser.add_argument("--resize-long-edge", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tta", choices=["none", "x4", "d4"], default="none")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--unsafe-load", action="store_true")
    return parser.parse_args()


def run_stage1_from_args(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device).eval()
    load_weights(model, args.checkpoint, device, args.unsafe_load)
    for param in model.parameters():
        param.requires_grad = False

    images = list_images(input_dir, args.recursive)
    print(f"Found {len(images)} images")
    saved = 0
    skipped = 0
    total_infer_time = 0.0
    timed_images = 0

    progress = tqdm(total=len(images), desc="Inference", unit="img")
    for batch_paths in batched(images, args.batch_size):
        pending = []
        tensors = []
        for path in batch_paths:
            rel_path = path.relative_to(input_dir) if args.recursive else Path(path.name)
            out_path = output_dir / rel_path
            if args.skip_existing and out_path.exists():
                skipped += 1
                progress.update(1)
                continue
            tensor, original_size = preprocess(path, args.resize_long_edge)
            pending.append((out_path, original_size))
            tensors.append(tensor)

        if not tensors:
            continue

        batch, model_sizes = pad_batch_to_common_size(tensors)
        batch = batch.to(device, non_blocking=True)

        synchronize_if_cuda(device)
        start_time = time.perf_counter()
        outputs = run_model(model, batch, args, device)
        synchronize_if_cuda(device)
        infer_time = time.perf_counter() - start_time
        infer_time_per_image = infer_time / len(tensors)
        total_infer_time += infer_time
        timed_images += len(tensors)

        for idx, (out_path, original_size) in enumerate(pending):
            save_output(outputs[idx], out_path, model_sizes[idx], original_size)
            saved += 1

        avg_time = total_infer_time / max(timed_images, 1)
        progress.update(len(pending))
        progress.set_postfix(
            saved=saved,
            skipped=skipped,
            time_per_img=f"{infer_time_per_image:.4f}s",
            avg=f"{avg_time:.4f}s",
        )

    progress.close()
    if timed_images:
        print(f"Average inference time per image: {total_infer_time / timed_images:.6f}s")
    print(f"Done. Output: {output_dir}")
    return output_dir



def run_stage1(
    input_dir,
    output_dir,
    checkpoint,
    batch_size=1,
    resize_long_edge=None,
    amp=False,
    tta="none",
    recursive=False,
    skip_existing=False,
    unsafe_load=False,
):
    args = SimpleNamespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        checkpoint=str(checkpoint),
        batch_size=batch_size,
        resize_long_edge=resize_long_edge,
        amp=amp,
        tta=tta,
        recursive=recursive,
        skip_existing=skip_existing,
        unsafe_load=unsafe_load,
    )
    return run_stage1_from_args(args)


def main():
    return run_stage1_from_args(parse_args())


if __name__ == "__main__":
    main()
