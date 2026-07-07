#!/usr/bin/env python
"""Run compact FUMO stage2 inference.

Inputs:
  - LQ image folder
  - P_int .npy folder with matching filenames
  - diffusion/controlnet weights
  - refine weights

Outputs:
  - prelim diffusion images, optional
  - final refined images
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from fumo_runtime import (
    apply_refine_residual,
    dtype_from_name,
    infer_diff_prelim,
    load_pipeline,
    load_prior_tensor,
    load_refine_models,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FUMO stage2 inference from LQ images, P_int priors, and trained weights.")
    parser.add_argument("--input_dir", required=True, help="Directory containing LQ images.")
    parser.add_argument("--prior_dir", required=True, help="Directory containing P_int .npy files.")
    parser.add_argument("--output_dir", required=True, help="Output root directory.")
    parser.add_argument("--run_name", default="fumo_stage2", help="Subdirectory name under output_dir.")
    parser.add_argument("--pretrained_model_name_or_path", required=True, help="Stable Diffusion 2.1/base model path or repo id.")
    parser.add_argument("--controlnet_dir", required=True, help="Trained ControlNet directory.")
    parser.add_argument("--unet_dir", required=True, help="Trained UNet directory.")
    parser.add_argument("--refine_net_path", required=True, help="Path to nafnet_refine.pth.")
    parser.add_argument("--refine_head_path", required=True, help="Path to nafnet_refine_head.pth.")
    parser.add_argument("--prompt", default="remove degradation")
    parser.add_argument("--resolution", type=int, default=768, help="Internal diffusion processing resolution.")
    parser.add_argument("--resolution_mode", default="full", choices=["square", "full"])
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--nafnet_width", type=int, default=64)
    parser.add_argument("--nafnet_middle_blk_num", type=int, default=1)
    parser.add_argument("--nafnet_enc_blk_nums", type=int, nargs="+", default=[1, 1, 1, 28])
    parser.add_argument("--nafnet_dec_blk_nums", type=int, nargs="+", default=[1, 1, 1, 1])
    parser.add_argument("--recursive", action="store_true", help="Search input images recursively and preserve subfolders.")
    parser.add_argument("--save_prelim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_input", action="store_true")
    parser.add_argument("--save_original_size", action="store_true", help="Resize outputs back to each original input size before saving.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--tta", default="none", choices=["none", "d4"], help="Diffusion TTA. d4 = 4 rotations x horizontal flip, averaged after inverse transform.")
    parser.add_argument("--refine_tta", default="none", choices=["none", "d4"], help="NAFNet refine TTA applied after diffusion prelim is averaged.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser.parse_args()


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def prior_path_for_image(image_path: Path, input_dir: Path, prior_dir: Path, recursive: bool) -> Path:
    if recursive:
        rel = image_path.relative_to(input_dir).with_suffix(".npy")
        nested = prior_dir / rel
        if nested.exists():
            return nested
    return prior_dir / f"{image_path.stem}.npy"


def output_relative_path(image_path: Path, input_dir: Path, recursive: bool) -> Path:
    if recursive:
        return image_path.relative_to(input_dir).with_suffix(".png")
    return Path(f"{image_path.stem}.png")


def shard_items(items: list[Path], num_shards: int, shard_id: int) -> list[Path]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < --num_shards")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_id]


def image_to_tensor(path: Path, resolution: int, resolution_mode: str, resize: tuple[int, int] | None):
    original = Image.open(path).convert("RGB")
    original_size = original.size
    if resolution_mode == "square":
        target_size = (resolution, resolution)
    elif resize is not None:
        target_size = resize
    else:
        target_size = original_size
    image = original.resize(target_size, Image.Resampling.BILINEAR) if original.size != target_size else original
    return TF.to_tensor(image).unsqueeze(0), original_size, original


def tensor_to_image(tensor: torch.Tensor, original_size: tuple[int, int] | None = None) -> Image.Image:
    image = TF.to_pil_image(tensor.detach().float().cpu().clamp(0.0, 1.0).squeeze(0))
    if original_size is not None and image.size != original_size:
        image = image.resize(original_size, Image.Resampling.BILINEAR)
    return image


def apply_d4_transform(tensor: torch.Tensor, rotation: int, hflip: bool) -> torch.Tensor:
    output = torch.rot90(tensor, k=rotation, dims=(-2, -1))
    if hflip:
        output = torch.flip(output, dims=(-1,))
    return output


def invert_d4_transform(tensor: torch.Tensor, rotation: int, hflip: bool) -> torch.Tensor:
    output = tensor
    if hflip:
        output = torch.flip(output, dims=(-1,))
    return torch.rot90(output, k=(-rotation) % 4, dims=(-2, -1))


def d4_variants() -> list[tuple[int, bool]]:
    return [(rotation, hflip) for rotation in range(4) for hflip in (False, True)]


def infer_diffusion_with_tta(pipeline, cond, prior, prompt: str, beta: float, tta: str):
    if tta == "none":
        prelim_pred = infer_diff_prelim(pipeline, cond, prior, prompt, beta)
        return ((prelim_pred + 1.0) / 2.0).clamp(0.0, 1.0).float()

    prelim_outputs = []
    for rotation, hflip in d4_variants():
        cond_aug = apply_d4_transform(cond, rotation, hflip)
        prior_aug = apply_d4_transform(prior, rotation, hflip)
        prelim_pred = infer_diff_prelim(pipeline, cond_aug, prior_aug, prompt, beta)
        prelim_aug = ((prelim_pred + 1.0) / 2.0).clamp(0.0, 1.0).float()
        prelim_outputs.append(invert_d4_transform(prelim_aug, rotation, hflip))
    return torch.stack(prelim_outputs, dim=0).mean(dim=0).clamp(0.0, 1.0)


def refine_with_tta(refine_net, refine_head, prelim, cond, prior, residual_scale: float, refine_tta: str):
    if refine_tta == "none":
        return apply_refine_residual(refine_net, refine_head, prelim, cond.float(), prior.float(), residual_scale)

    refined_outputs = []
    for rotation, hflip in d4_variants():
        prelim_aug = apply_d4_transform(prelim, rotation, hflip)
        cond_aug = apply_d4_transform(cond, rotation, hflip)
        prior_aug = apply_d4_transform(prior, rotation, hflip)
        refined_aug = apply_refine_residual(
            refine_net,
            refine_head,
            prelim_aug,
            cond_aug.float(),
            prior_aug.float(),
            residual_scale,
        )
        refined_outputs.append(invert_d4_transform(refined_aug, rotation, hflip))
    return torch.stack(refined_outputs, dim=0).mean(dim=0).clamp(0.0, 1.0)


def infer_with_tta(pipeline, refine_net, refine_head, cond, prior, prompt: str, beta: float, residual_scale: float, tta: str, refine_tta: str):
    prelim = infer_diffusion_with_tta(pipeline, cond, prior, prompt, beta, tta)
    refined = refine_with_tta(refine_net, refine_head, prelim, cond, prior, residual_scale, refine_tta)
    return prelim, refined


def save_metadata(run_root: Path, args: argparse.Namespace, image_count: int) -> None:
    payload = vars(args).copy()
    payload["image_count"] = image_count
    with open(run_root / "inference_config.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def run_stage2_from_args(args):
    input_dir = Path(args.input_dir).resolve()
    prior_dir = Path(args.prior_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    run_root = output_root / args.run_name if args.run_name else output_root
    if args.run_name:
        prelim_dir = run_root / "prelim"
        final_dir = run_root / "final"
        input_save_dir = run_root / "input"
    else:
        prelim_dir = run_root / "_prelim"
        final_dir = run_root
        input_save_dir = run_root / "_input"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"Prior directory does not exist: {prior_dir}")

    images = shard_items(list_images(input_dir, args.recursive), args.num_shards, args.shard_id)
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    for directory, enabled in ((prelim_dir, args.save_prelim), (final_dir, args.save_final), (input_save_dir, args.save_input)):
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    save_metadata(run_root, args, len(images))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_name(args.dtype)
    pipeline_args = SimpleNamespace(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        controlnet_dir=args.controlnet_dir,
        unet_dir=args.unet_dir,
        resolution=args.resolution,
    )

    print(f"Input: {input_dir}")
    print(f"Prior: {prior_dir}")
    print(f"Output: {run_root}")
    print(f"Images: {len(images)} shard {args.shard_id}/{args.num_shards}")
    print(f"Diffusion TTA: {args.tta}")
    print(f"Refine TTA: {args.refine_tta}")
    print(f"Device: {device} dtype={args.dtype}")

    pipeline = load_pipeline(pipeline_args, device, dtype)
    refine_net, refine_head = load_refine_models(args, device)

    missing_priors = []
    processed = 0
    skipped = 0
    start = time.perf_counter()
    with torch.no_grad():
        for image_path in tqdm(images, desc="FUMO stage2 inference"):
            prior_path = prior_path_for_image(image_path, input_dir, prior_dir, args.recursive)
            rel = output_relative_path(image_path, input_dir, args.recursive)
            final_path = final_dir / rel
            prelim_path = prelim_dir / rel
            input_path = input_save_dir / rel

            if not prior_path.exists():
                missing_priors.append(str(image_path))
                continue
            if args.skip_existing and args.save_final and final_path.exists():
                skipped += 1
                continue

            cond, original_size, original_image = image_to_tensor(
                image_path,
                args.resolution,
                args.resolution_mode,
                tuple(args.resize) if args.resize is not None else None,
            )
            prior = load_prior_tensor(prior_path)
            if prior.shape[-2:] != cond.shape[-2:]:
                prior = torch.nn.functional.interpolate(prior, size=cond.shape[-2:], mode="bilinear", align_corners=False)

            cond = cond.to(device=device)
            prior = prior.to(device=device)
            prelim, refined = infer_with_tta(
                pipeline,
                refine_net,
                refine_head,
                cond,
                prior,
                args.prompt,
                args.beta,
                args.residual_scale,
                args.tta,
                args.refine_tta,
            )

            save_size = original_size if args.save_original_size else None
            if args.save_prelim:
                prelim_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(prelim, save_size).save(prelim_path)
            if args.save_final:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(refined, save_size).save(final_path)
            if args.save_input:
                input_path.parent.mkdir(parents=True, exist_ok=True)
                original_image.save(input_path)
            processed += 1

    if missing_priors:
        with open(run_root / "missing_priors.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(missing_priors) + "\n")

    elapsed = time.perf_counter() - start
    seconds_per_image = elapsed / max(processed, 1)
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Missing priors: {len(missing_priors)}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Seconds/image: {seconds_per_image:.3f}")
    return final_dir



def run_stage2(
    input_dir,
    prior_dir,
    output_dir,
    run_name="",
    pretrained_model_name_or_path="Manojb/stable-diffusion-2-1-base",
    controlnet_dir="weights/stage2/controlnet",
    unet_dir="weights/stage2/unet",
    refine_net_path="weights/stage2/nafnet_refine.pth",
    refine_head_path="weights/stage2/nafnet_refine_head.pth",
    prompt="remove degradation",
    resolution=768,
    resolution_mode="full",
    resize=None,
    beta=0.25,
    residual_scale=0.1,
    device="cuda",
    dtype="bf16",
    nafnet_width=64,
    nafnet_middle_blk_num=1,
    nafnet_enc_blk_nums=None,
    nafnet_dec_blk_nums=None,
    recursive=False,
    save_prelim=True,
    save_final=True,
    save_input=False,
    save_original_size=False,
    skip_existing=False,
    tta="none",
    refine_tta="none",
    limit=None,
    num_shards=1,
    shard_id=0,
):
    args = SimpleNamespace(
        input_dir=str(input_dir),
        prior_dir=str(prior_dir),
        output_dir=str(output_dir),
        run_name=run_name,
        pretrained_model_name_or_path=str(pretrained_model_name_or_path),
        controlnet_dir=str(controlnet_dir),
        unet_dir=str(unet_dir),
        refine_net_path=str(refine_net_path),
        refine_head_path=str(refine_head_path),
        prompt=prompt,
        resolution=resolution,
        resolution_mode=resolution_mode,
        resize=resize,
        beta=beta,
        residual_scale=residual_scale,
        device=device,
        dtype=dtype,
        nafnet_width=nafnet_width,
        nafnet_middle_blk_num=nafnet_middle_blk_num,
        nafnet_enc_blk_nums=nafnet_enc_blk_nums or [1, 1, 1, 28],
        nafnet_dec_blk_nums=nafnet_dec_blk_nums or [1, 1, 1, 1],
        recursive=recursive,
        save_prelim=save_prelim,
        save_final=save_final,
        save_input=save_input,
        save_original_size=save_original_size,
        skip_existing=skip_existing,
        tta=tta,
        refine_tta=refine_tta,
        limit=limit,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    return run_stage2_from_args(args)


def main() -> None:
    return run_stage2_from_args(parse_args())


if __name__ == "__main__":
    main()
