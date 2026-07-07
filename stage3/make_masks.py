#!/usr/bin/env python3
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build raindrop and reflection masks from paired images."
    )
    parser.add_argument("--root", type=Path, default=Path("data/RDRF_dataset"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument(
        "--mask-types",
        nargs="+",
        default=["raindrop", "reflection"],
        help="Mask types to build: raindrop, reflection, rain_drop, or all.",
    )
    parser.add_argument("--lq-dir", default="LQ")
    parser.add_argument("--is-dir", default="IS")
    parser.add_argument("--is2-dir", default="IS2")
    parser.add_argument("--drop-mask-dir", default="DROP_MASK")
    parser.add_argument("--drop-vis-dir", default="DROP_MASK_VIS")
    parser.add_argument("--reflection-mask-dir", default="REFLECTION_MASK")
    parser.add_argument("--reflection-vis-dir", default="REFLECTION_MASK_VIS")
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Backward-compatible override for raindrop mask output directory.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Backward-compatible override for raindrop visualization output directory.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Append a suffix to mask/visualization output directories, e.g. "
            "--output-suffix _OTSU writes DROP_MASK_OTSU and REFLECTION_MASK_OTSU."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help=(
            "Fixed difference threshold. Values <= 1 are interpreted as 0-1 scale; "
            "otherwise 0-255 scale."
        ),
    )
    parser.add_argument(
        "--raindrop-threshold",
        type=float,
        default=None,
        help="Optional fixed threshold override for DROP_MASK.",
    )
    parser.add_argument(
        "--reflection-threshold",
        type=float,
        default=None,
        help="Optional fixed threshold override for REFLECTION_MASK.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "percentile", "otsu"],
        default="fixed",
        help=(
            "How to choose the binary threshold. fixed uses --threshold or per-mask overrides; "
            "percentile thresholds each image by --threshold-percentile; otsu uses Otsu's method."
        ),
    )
    parser.add_argument(
        "--otsu",
        action="store_true",
        help="Shortcut for --threshold-mode otsu. Builds binary masks with per-image Otsu thresholding.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile used when --threshold-mode percentile.",
    )
    parser.add_argument(
        "--diff-mode",
        choices=["abs", "positive", "negative"],
        default="positive",
        help=(
            "abs: |first-second|, positive: max(first-second, 0), negative: max(second-first, 0). "
            "Raindrop uses first=LQ, second=IS. Reflection uses first=IS, second=IS2."
        ),
    )
    parser.add_argument(
        "--reduce",
        choices=["luma", "mean", "max"],
        default="luma",
        help="How RGB differences are reduced to one mask score.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=0.0,
        help="Optional Gaussian blur on the diff score before thresholding.",
    )
    parser.add_argument(
        "--median-size",
        type=int,
        default=3,
        help="Optional odd-sized median filter for binary mask cleanup. Set 0 or 1 to disable.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save float32 masks in 0/1 format as .npy files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask and visualization files.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Process only the first N matched images in each split, useful for quick checks.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker processes. Use 0 to use all CPU cores.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Chunk size for multiprocessing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict-inputs",
        action="store_true",
        help="Raise an error when a requested split/input directory is missing instead of skipping it.",
    )
    return parser.parse_args()


def list_images_by_stem(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    paths = collect_images_by_stem(directory)
    if not paths:
        raise ValueError(f"No images found in: {directory}")
    return paths


def collect_images_by_stem(directory):
    paths = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
        paths[path.stem] = path
    return paths


def load_rgb(path):
    return Image.open(path).convert("RGB")


def compute_diff_score(first_image, second_image, diff_mode, reduce):
    first = np.asarray(first_image, dtype=np.float32)
    second = np.asarray(second_image, dtype=np.float32)

    if diff_mode == "abs":
        diff = np.abs(first - second)
    elif diff_mode == "positive":
        diff = np.maximum(first - second, 0.0)
    else:
        diff = np.maximum(second - first, 0.0)

    if reduce == "mean":
        score = diff.mean(axis=2)
    elif reduce == "max":
        score = diff.max(axis=2)
    else:
        score = 0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]

    return np.clip(score, 0.0, 255.0)


def normalize_fixed_threshold(threshold):
    if threshold <= 1.0:
        threshold = threshold * 255.0
    return float(np.clip(threshold, 0.0, 255.0))


def otsu_threshold(score_array):
    values = np.clip(score_array.round(), 0, 255).astype(np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0

    bins = np.arange(256, dtype=np.float64)
    sum_total = float((bins * hist).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    max_between = -1.0
    threshold = 0.0

    for index in range(256):
        weight_bg += hist[index]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += index * hist[index]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = float(index)
    return threshold


def resolve_threshold(score_array, options):
    mode = options["threshold_mode"]
    if mode == "fixed":
        return normalize_fixed_threshold(options["threshold"])
    if mode == "percentile":
        percentile = float(options["threshold_percentile"])
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("--threshold-percentile must be in [0, 100]")
        return float(np.percentile(score_array, percentile))
    if mode == "otsu":
        return otsu_threshold(score_array)
    raise ValueError(f"Unsupported threshold mode: {mode}")


def make_binary_mask(score, options):
    score_image = Image.fromarray(score.round().astype(np.uint8), mode="L")
    if options["blur_radius"] > 0:
        score_image = score_image.filter(ImageFilter.GaussianBlur(radius=options["blur_radius"]))

    score_array = np.asarray(score_image, dtype=np.float32)
    threshold = resolve_threshold(score_array, options)
    if options["threshold_mode"] == "otsu" and threshold <= 0.0:
        mask = np.zeros_like(score_array, dtype=np.uint8)
    else:
        mask = (score_array > threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")

    median_size = options["median_size"]
    if median_size and median_size > 1:
        if median_size % 2 == 0:
            raise ValueError("--median-size must be odd")
        mask_image = mask_image.filter(ImageFilter.MedianFilter(size=median_size))

    return mask_image, threshold


def colorize_diff(score):
    normalized = np.clip(score / max(float(score.max()), 1.0), 0.0, 1.0)
    heat = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(255 * normalized * 1.6, 0, 255)
    heat[..., 1] = np.clip(255 * (1.0 - np.abs(normalized - 0.5) * 2.0), 0, 255)
    heat[..., 2] = np.clip(255 * (1.0 - normalized) * 0.7, 0, 255)
    return Image.fromarray(heat, mode="RGB")


def make_overlay(base_image, mask_image):
    base = base_image.convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 0, 0, 120))
    alpha = mask_image.point(lambda value: 120 if value > 0 else 0)
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


def add_label(image, label):
    labeled = ImageOps.expand(image.convert("RGB"), border=(0, 28, 0, 0), fill=(20, 20, 20))
    draw = ImageDraw.Draw(labeled)
    draw.text((8, 7), label, fill=(255, 255, 255))
    return labeled


def make_visualization(
    first_image,
    second_image,
    score,
    mask_image,
    first_label,
    second_label,
    mask_label,
    threshold_label,
):
    diff_image = colorize_diff(score).resize(first_image.size, Image.Resampling.BILINEAR)
    overlay = make_overlay(first_image, mask_image)
    panels = [
        add_label(first_image, first_label),
        add_label(second_image, second_label),
        add_label(diff_image, "diff"),
        add_label(mask_image.convert("RGB"), threshold_label),
        add_label(overlay, "overlay"),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def process_mask_item(task):
    stem, first_path, second_path, mask_path, vis_path, npy_path, options = task
    first_path = Path(first_path)
    second_path = Path(second_path)
    mask_path = Path(mask_path)
    vis_path = Path(vis_path)
    npy_path = Path(npy_path)

    if (
        not options["overwrite"]
        and mask_path.exists()
        and vis_path.exists()
        and (not options["save_npy"] or npy_path.exists())
    ):
        return "skipped"

    first_image = load_rgb(first_path)
    second_image = load_rgb(second_path)
    if first_image.size != second_image.size:
        raise ValueError(
            f"Size mismatch for {stem}: "
            f"{options['first_label']}={first_image.size}, {options['second_label']}={second_image.size}"
        )

    score = compute_diff_score(
        first_image,
        second_image,
        options["diff_mode"],
        options["reduce"],
    )
    mask_image, threshold = make_binary_mask(score, options)
    mask_image.save(mask_path)

    if options["save_npy"]:
        np.save(npy_path, (np.asarray(mask_image, dtype=np.float32) / 255.0))

    visualization = make_visualization(
        first_image,
        second_image,
        score,
        mask_image,
        options["first_label"],
        options["second_label"],
        options["mask_label"],
        format_threshold_label(options["mask_label"], options["threshold_mode"], threshold),
    )
    visualization.save(vis_path)
    return "written"


def format_threshold_label(mask_label, threshold_mode, threshold):
    return f"{mask_label} t={threshold:.1f} {threshold_mode}"


def count_statuses(statuses):
    counts = {"written": 0, "skipped": 0}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def make_stats(matched=0, written=0, skipped=0, elapsed_seconds=0.0, source_keys=None):
    return {
        "matched": int(matched),
        "written": int(written),
        "skipped": int(skipped),
        "elapsed_seconds": float(elapsed_seconds),
        "source_keys": set(source_keys or ()),
    }


def add_stats(total, update):
    total["matched"] += update["matched"]
    total["written"] += update["written"]
    total["skipped"] += update["skipped"]
    total["elapsed_seconds"] += update["elapsed_seconds"]
    total["source_keys"].update(update["source_keys"])
    return total


def normalize_mask_types(mask_types):
    aliases = {
        "all": ("raindrop", "reflection"),
        "drop": ("raindrop",),
        "rain": ("raindrop",),
        "rain_drop": ("raindrop",),
        "raindrop": ("raindrop",),
        "reflection": ("reflection",),
        "reflect": ("reflection",),
    }

    normalized = []
    for mask_type in mask_types:
        key = str(mask_type).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unsupported mask type: {mask_type}")
        for canonical in aliases[key]:
            if canonical not in normalized:
                normalized.append(canonical)
    return normalized


def build_mask_specs(args):
    drop_mask_dir = args.mask_dir if args.mask_dir is not None else args.drop_mask_dir
    drop_vis_dir = args.vis_dir if args.vis_dir is not None else args.drop_vis_dir
    specs = {
        "raindrop": {
            "name": "raindrop",
            "first_dir": args.lq_dir,
            "second_dir": args.is_dir,
            "first_label": "LQ",
            "second_label": "IS",
            "mask_label": "DROP_MASK",
            "mask_dir": append_output_suffix(drop_mask_dir, args.output_suffix),
            "vis_dir": append_output_suffix(drop_vis_dir, args.output_suffix),
            "threshold": args.raindrop_threshold,
        },
        "reflection": {
            "name": "reflection",
            "first_dir": args.is_dir,
            "second_dir": args.is2_dir,
            "first_label": "IS",
            "second_label": "IS2",
            "mask_label": "REFLECTION_MASK",
            "mask_dir": append_output_suffix(args.reflection_mask_dir, args.output_suffix),
            "vis_dir": append_output_suffix(args.reflection_vis_dir, args.output_suffix),
            "threshold": args.reflection_threshold,
        },
    }
    return [specs[mask_type] for mask_type in normalize_mask_types(args.mask_types)]


def append_output_suffix(directory, suffix):
    suffix = str(suffix or "")
    if not suffix:
        return directory
    path = Path(directory)
    return str(path.with_name(f"{path.name}{suffix}"))


def process_split(args, split):
    split_stats = make_stats()
    split_root = args.root / split
    if not split_root.exists():
        message = f"[{split}] split directory does not exist: {split_root}"
        if args.strict_inputs:
            raise FileNotFoundError(message)
        print(f"{message}; skipped")
        return split_stats
    for spec in build_mask_specs(args):
        add_stats(split_stats, process_mask_split(args, split, split_root, spec))
    return split_stats


def process_mask_split(args, split, split_root, spec):
    first_dir = split_root / spec["first_dir"]
    second_dir = split_root / spec["second_dir"]
    first_paths = get_images_by_stem_or_skip(first_dir, args.strict_inputs, split, spec["first_label"])
    second_paths = get_images_by_stem_or_skip(second_dir, args.strict_inputs, split, spec["second_label"])
    if first_paths is None or second_paths is None:
        print(
            f"[{split}] {spec['mask_label']} skipped; "
            f"needs {spec['first_label']}={first_dir} and {spec['second_label']}={second_dir}"
        )
        return make_stats()

    mask_dir = split_root / spec["mask_dir"]
    vis_dir = split_root / spec["vis_dir"]
    matched = []
    missing = []
    for stem, first_path in first_paths.items():
        second_path = second_paths.get(stem)
        if second_path is None:
            missing.append(first_path.name)
            continue
        matched.append((stem, first_path, second_path))

    if args.max_items is not None:
        matched = matched[: args.max_items]

    print(
        f"[{split}] {spec['mask_label']} matched={len(matched)} "
        f"missing_{spec['second_label']}={len(missing)} output={mask_dir}"
    )
    if missing:
        print(f"[{split}] first missing {spec['second_label']} files: {', '.join(missing[:5])}")

    if args.dry_run:
        return make_stats(
            matched=len(matched),
            source_keys={f"{split}/{stem}" for stem, _, _ in matched},
        )

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")

    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "threshold": args.threshold,
        "diff_mode": args.diff_mode,
        "reduce": args.reduce,
        "blur_radius": args.blur_radius,
        "median_size": args.median_size,
        "save_npy": args.save_npy,
        "overwrite": args.overwrite,
        "first_label": spec["first_label"],
        "second_label": spec["second_label"],
        "mask_label": spec["mask_label"],
        "threshold_mode": args.threshold_mode,
        "threshold_percentile": args.threshold_percentile,
    }
    threshold_override = spec.get("threshold")
    options["threshold"] = args.threshold if threshold_override is None else threshold_override
    tasks = []
    for stem, first_path, second_path in matched:
        mask_path = mask_dir / f"{stem}.png"
        vis_path = vis_dir / f"{stem}.png"
        npy_path = mask_dir / f"{stem}.npy"
        tasks.append((stem, first_path, second_path, mask_path, vis_path, npy_path, options))

    start_time = time.perf_counter()
    worker_count = os.cpu_count() if args.num_workers == 0 else args.num_workers
    if worker_count is None:
        worker_count = 1
    worker_count = max(1, worker_count)

    if worker_count == 1:
        statuses = [
            process_mask_item(task)
            for task in tqdm(tasks, desc=f"{split} {spec['mask_label']}")
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            statuses = list(
                tqdm(
                    executor.map(
                        process_mask_item,
                        tasks,
                        chunksize=args.chunksize,
                    ),
                    total=len(tasks),
                    desc=f"{split} {spec['mask_label']} x{worker_count}",
                )
            )

    elapsed_seconds = time.perf_counter() - start_time
    counts = count_statuses(statuses)
    print(
        f"[{split}] {spec['mask_label']} "
        f"written={counts.get('written', 0)} skipped={counts.get('skipped', 0)}"
    )
    return make_stats(
        matched=len(matched),
        written=counts.get("written", 0),
        skipped=counts.get("skipped", 0),
        elapsed_seconds=elapsed_seconds,
        source_keys={f"{split}/{stem}" for stem, _, _ in matched},
    )


def get_images_by_stem_or_skip(directory, strict_inputs, split, label):
    try:
        return list_images_by_stem(directory)
    except (FileNotFoundError, ValueError) as error:
        if strict_inputs:
            raise
        print(f"[{split}] {label} input unavailable: {error}")
        return None


def main():
    args = parse_args()
    if args.otsu:
        args.threshold_mode = "otsu"
    total_stats = make_stats()
    total_start_time = time.perf_counter()
    for split in args.splits:
        add_stats(total_stats, process_split(args, split))
    total_wall_seconds = time.perf_counter() - total_start_time
    print_runtime_summary(total_stats, total_wall_seconds)


def print_runtime_summary(stats, wall_seconds):
    matched = stats["matched"]
    source_images = len(stats["source_keys"])
    elapsed_seconds = stats["elapsed_seconds"]
    summary_parts = [
        "Mask generation runtime:",
        f"source_images={source_images}",
        f"mask_images={matched}",
        f"written={stats['written']}",
        f"skipped={stats['skipped']}",
        f"worker_elapsed={elapsed_seconds:.2f}s",
        f"wall_elapsed={wall_seconds:.2f}s",
    ]
    if source_images > 0:
        summary_parts.append(f"seconds_per_image={wall_seconds / source_images:.4f}")
    if matched > 0 and elapsed_seconds > 0:
        summary_parts.append(f"seconds_per_mask_image={elapsed_seconds / matched:.4f}")
    print(" ".join(summary_parts))


if __name__ == "__main__":
    main()
