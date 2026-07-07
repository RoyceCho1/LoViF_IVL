#!/usr/bin/env python
"""Generate FUMO P_int priors as .npy files only.

This script processes images in one directory and saves one float32 .npy prior per image.
It intentionally does not write visualization PNGs.
"""

import argparse
import multiprocessing as mp
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError("qwen-vl-utils is required. Install with: pip install qwen-vl-utils") from exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CANDIDATES = ["None", "Minor", "Mid", "Major", "Critical"]
WEIGHTS = {"None": 1, "Minor": 2, "Mid": 3, "Major": 4, "Critical": 5}

BOOST_FACTOR = 1.5
BOOST_CAP = 3.8
GUIDED_FILTER_EPS = 0.01**2
KSIZE_PREBLUR = 23
FIXED_MIN_VAL = 1.0
FIXED_MAX_VAL = 4.0
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P_int .npy priors with Qwen-VL.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input LQ images.")
    parser.add_argument("--output_dir", required=True, help="Directory where .npy P_int files will be saved.")
    parser.add_argument("--model_path", required=True, help="Local Qwen-VL model directory or Hugging Face repo id.")
    parser.add_argument("--model_family", default="qwen3", choices=["qwen2.5", "qwen3"])
    parser.add_argument("--recursive", action="store_true", help="Search input images recursively and preserve subfolders.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate priors even if .npy files already exist.")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of GPU worker processes.")
    parser.add_argument("--max_long_side", type=int, default=99999, help="Downscale images whose long side exceeds this value.")
    return parser.parse_args()


def get_model_class(model_family: str):
    if model_family == "qwen2.5":
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Qwen2_5_VLForConditionalGeneration is unavailable in this transformers install.")
        return Qwen2_5_VLForConditionalGeneration
    if model_family == "qwen3":
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError("Qwen3VLForConditionalGeneration is unavailable in this transformers install.")
        return Qwen3VLForConditionalGeneration
    raise ValueError(f"Unsupported model_family: {model_family}")


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def output_path_for_image(image_path: Path, input_dir: Path, output_dir: Path, recursive: bool) -> Path:
    if recursive:
        rel = image_path.relative_to(input_dir).with_suffix(".npy")
        return output_dir / rel
    return output_dir / f"{image_path.stem}.npy"


def get_candidate_ids(tokenizer) -> dict[str, int]:
    ids_map = {}
    for word in CANDIDATES:
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            ids_map[word] = ids[0]
        else:
            ids2 = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
            ids_map[word] = ids2[0] if len(ids2) == 1 else ids[0]
    return ids_map


def choose_patch_params(height: int, width: int) -> tuple[str, int, float]:
    max_side = max(height, width)
    if max_side > 1900:
        return "nonscale", 200, 1.0
    if 1000 < max_side <= 1900:
        return "nonscale", 120, 1.0
    if 750 < max_side <= 1000:
        return "nonscale", 100, 1.0
    return "nonscale", 80, 1.0


@torch.no_grad()
def score_only(pil_img: Image.Image, model, processor, candidate_ids: dict[str, int], device: str) -> float:
    prompt = (
        "For this image patch, evaluate the severity of reflection. "
        "Use exactly one of the following words: None, Minor, Mid, Major, Critical. "
        "Reply with only the chosen word, with no extra text."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    inputs = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True)
    logits = out.scores[0][0]
    sel_logits = torch.stack([logits[candidate_ids[word]] for word in CANDIDATES], dim=0)
    probs = F.softmax(sel_logits, dim=-1)
    weights = torch.tensor([WEIGHTS[word] for word in CANDIDATES], device=probs.device, dtype=probs.dtype)
    return float(torch.sum(probs * weights).item())


@torch.no_grad()
def get_reflection_bounding_boxes(image_bgr: np.ndarray, model, processor, device: str) -> list[tuple[int, int, int, int]]:
    height, width = image_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    prompt = (
        f"This is an analysis task. The image dimensions are {width}x{height} pixels.\n"
        "In the provided image, please identify and locate any reflections, ghosting, double images, artifacts, "
        "or unnatural light spots and streaks within the image.\n\n"
        "Here are the rules for the output:\n"
        "1. If a single, large, and contiguous area is covered by reflection, please provide one large bounding box.\n"
        "2. If there are multiple, separate, non-contiguous reflection areas, please provide a unique bounding box.\n"
        "3. Ensure boxes are accurate without including non-reflection parts.\n\n"
        "The required output format is a list of lists of four integers, e.g.: [[x1, y1, x2, y2]].\n"
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    raw_response = processor.decode(response_ids, skip_special_tokens=True).strip()
    matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", raw_response)
    return [tuple(map(int, match)) for match in matches]


def create_heatmap_for_image(image_path: Path, prior_path: Path, model, processor, candidate_ids: dict[str, int], device: str, max_long_side: int) -> None:
    image_raw = cv2.imread(str(image_path))
    if image_raw is None:
        print(f"[{device}] Warning: failed to read image {image_path}")
        return

    height0, width0 = image_raw.shape[:2]
    max_side = max(height0, width0)
    if max_side > max_long_side:
        scale0 = max_long_side / max_side
        image_for_proc = cv2.resize(image_raw, (int(width0 * scale0), int(height0 * scale0)), interpolation=cv2.INTER_AREA)
    else:
        image_for_proc = image_raw

    height_proc, width_proc = image_for_proc.shape[:2]
    mode, patch_size, scale = choose_patch_params(height_proc, width_proc)
    if mode == "resize":
        image_proc = cv2.resize(image_for_proc, (int(width_proc * scale), int(height_proc * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        image_proc = image_for_proc
    height, width = image_proc.shape[:2]

    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    image_pad = cv2.copyMakeBorder(image_proc, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    padded_h, padded_w = image_pad.shape[:2]
    rows, cols = padded_h // patch_size, padded_w // patch_size

    score_grid = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            patch = image_pad[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
            patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            score_grid[row, col] = score_only(patch_pil, model, processor, candidate_ids, device)

    score_field = np.zeros((padded_h, padded_w), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            score_field[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = score_grid[row, col]
    score_field = score_field[:height, :width]

    boxes = get_reflection_bounding_boxes(image_proc, model, processor, device)
    enhanced = score_field.copy()
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        enhanced[y1:y2, x1:x2] = np.minimum(enhanced[y1:y2, x1:x2] * BOOST_FACTOR, BOOST_CAP)

    radius = int(patch_size * 1.5)
    guide_base = cv2.cvtColor(image_proc, cv2.COLOR_BGR2GRAY)
    guide = cv2.GaussianBlur(guide_base, (KSIZE_PREBLUR, KSIZE_PREBLUR), 0)
    smoothed = cv2.ximgproc.guidedFilter(guide=guide, src=enhanced, radius=radius, eps=GUIDED_FILTER_EPS)

    clipped = np.clip(smoothed, FIXED_MIN_VAL, FIXED_MAX_VAL)
    norm = (clipped - FIXED_MIN_VAL) / max(FIXED_MAX_VAL - FIXED_MIN_VAL, EPS)
    final_h, final_w = image_for_proc.shape[:2]
    norm_resized = cv2.resize(norm, (final_w, final_h), interpolation=cv2.INTER_LINEAR)

    prior_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prior_path, norm_resized.astype(np.float32))
    if "cuda" in str(device):
        torch.cuda.empty_cache()


def worker(tasks: list[tuple[str, str]], device_id: int, model_path: str, model_family: str, max_long_side: int) -> None:
    device = f"cuda:{device_id}"
    print(f"Worker started on {device}")
    try:
        model_cls = get_model_class(model_family)
        model = model_cls.from_pretrained(model_path, dtype="auto", device_map={"": device})
        processor = AutoProcessor.from_pretrained(model_path)
        candidate_ids = get_candidate_ids(processor.tokenizer)
        print(f"Model loaded on {device}")
    except Exception as exc:
        print(f"[{device}] Failed to load model: {exc}")
        raise

    for image_path, prior_path in tqdm(tasks, desc=f"Worker {device_id}"):
        try:
            create_heatmap_for_image(Path(image_path), Path(prior_path), model, processor, candidate_ids, device, max_long_side)
        except Exception as exc:
            print(f"[{device}] Error processing {image_path}: {exc}")


def chunk_tasks(tasks: list[tuple[str, str]], num_chunks: int) -> list[list[tuple[str, str]]]:
    chunks = [[] for _ in range(num_chunks)]
    for idx, task in enumerate(tasks):
        chunks[idx % num_chunks].append(task)
    return chunks


def run_p_int_from_args(args):
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found. P_int generation expects GPU inference.")
    if args.max_workers is not None:
        num_gpus = min(num_gpus, args.max_workers)

    image_paths = list_images(input_dir, args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    tasks = []
    for image_path in image_paths:
        prior_path = output_path_for_image(image_path, input_dir, output_dir, args.recursive)
        if args.overwrite or not prior_path.exists():
            tasks.append((str(image_path), str(prior_path)))

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model_path} ({args.model_family})")
    print(f"Images: {len(image_paths)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Workers: {num_gpus}")

    if not tasks:
        print("All P_int files already exist.")
        return output_dir

    if num_gpus == 1:
        print("Execution: single process on cuda:0")
        worker(tasks, 0, args.model_path, args.model_family, args.max_long_side)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for device_id, task_chunk in enumerate(chunk_tasks(tasks, num_gpus)):
            if not task_chunk:
                continue
            process = mp.Process(
                target=worker,
                args=(task_chunk, device_id, args.model_path, args.model_family, args.max_long_side),
            )
            process.start()
            processes.append(process)

        failed = False
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed = True
        if failed:
            raise RuntimeError("One or more P_int workers failed.")

    missing_outputs = [prior_path for _, prior_path in tasks if not Path(prior_path).exists()]
    if missing_outputs:
        preview = ", ".join(str(path) for path in missing_outputs[:5])
        raise RuntimeError(f"P_int generation missed {len(missing_outputs)} output files. First missing: {preview}")

    print("Done.")
    return output_dir



def run_p_int(
    input_dir,
    output_dir,
    model_path="Qwen/Qwen3-VL-8B-Instruct",
    model_family="qwen3",
    recursive=False,
    overwrite=False,
    max_workers=1,
    max_long_side=99999,
):
    args = SimpleNamespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        model_path=str(model_path),
        model_family=model_family,
        recursive=recursive,
        overwrite=overwrite,
        max_workers=max_workers,
        max_long_side=max_long_side,
    )
    return run_p_int_from_args(args)


def main() -> None:
    mp.set_start_method("spawn", force=True)
    return run_p_int_from_args(parse_args())


if __name__ == "__main__":
    main()
