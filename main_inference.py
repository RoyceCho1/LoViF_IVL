#!/usr/bin/env python
"""Run the FUMO final inference pipeline.

Pipeline:
  LQ -> stage1 IS -> Qwen3-VL P_INT -> stage2 IS2 -> make masks -> stage3 blending -> result
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent
STAGE1_DIR = FINAL_DIR / "stage1"
P_INT_CODE_DIR = FINAL_DIR / "stage2" / "P_INT"
STAGE2_CODE_DIR = FINAL_DIR / "stage2" / "inference"
STAGE3_DIR = FINAL_DIR / "stage3"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class TimingSummary:
    def __init__(self) -> None:
        self.records = []

    def add(self, name: str, image_count: int, elapsed: float) -> None:
        self.records.append((name, image_count, elapsed))

    def print_total(self, elapsed: float) -> None:
        print("\n[timing] summary")
        total_image_count = max((image_count for _, image_count, _ in self.records), default=0)
        for name, image_count, stage_elapsed in self.records:
            seconds_per_image = stage_elapsed / max(image_count, 1)
            print(
                f"[timing] {name}: images={image_count} "
                f"elapsed={stage_elapsed:.2f}s sec/image={seconds_per_image:.3f}s"
            )
        total_seconds_per_image = elapsed / max(total_image_count, 1)
        print(
            f"[timing] total: images={total_image_count} "
            f"elapsed={elapsed:.2f}s sec/image={total_seconds_per_image:.3f}s"
        )


SETTINGS = {
    "run_stages": ["stage1", "p_int", "stage2", "make_mask", "stage3"],
    "cuda_visible_devices": "0",
    "recursive": False,
    "skip_existing": False,
    "paths": {
        "data_root": "data",
        "split": "test",
        "lq_dir": "data/test/LQ",
        "is_dir": "data/test/IS",
        "p_int_dir": "data/test/P_INT",
        "is2_dir": "data/test/IS2",
        "drop_mask_dir": "data/test/drop_mask",
        "reflection_mask_dir": "data/test/reflection_mask",
        "result_dir": "result",
    },
    "stage1": {
        "checkpoint": "weights/stage1/best_ckpt",
        "batch_size": 1,
        "resize_long_edge": None,
        "amp": False,
        "tta": "x4",
        "unsafe_load": False,
    },
    "p_int": {
        "model_path": "Qwen/Qwen3-VL-8B-Instruct",
        "model_family": "qwen3",
        "max_workers": 1,
        "max_long_side": 99999,
        "overwrite": False,
    },
    "stage2": {
        "run_name": "",
        "pretrained_model_name_or_path": "Manojb/stable-diffusion-2-1-base",
        "controlnet_dir": "weights/stage2/controlnet",
        "unet_dir": "weights/stage2/unet",
        "refine_net_path": "weights/stage2/nafnet_refine.pth",
        "refine_head_path": "weights/stage2/nafnet_refine_head.pth",
        "prompt": "remove degradation",
        "resolution": 768,
        "resolution_mode": "full",
        "resize": None,
        "nafnet_width": 96,
        "nafnet_middle_blk_num": 4,
        "nafnet_enc_blk_nums": [1, 1, 1, 32],
        "nafnet_dec_blk_nums": [1, 1, 1, 1],
        "beta": 0.25,
        "residual_scale": 0.1,
        "device": "cuda",
        "dtype": "bf16",
        "tta": "d4",
        "refine_tta": "d4",
        "save_prelim": False,
        "save_final": True,
        "save_input": False,
        "save_original_size": False,
        "limit": None,
        "num_shards": 1,
        "shard_id": 0,
    },
    "make_mask": {
        "num_workers": 16,
        "threshold_mode": "auto",
        "threshold": 20.0,
        "threshold_percentile": 95.0,
        "median_size": 3,
        "overwrite": False,
        "chunksize": 8,
    },
    "stage3": {
        "checkpoint": "weights/stage3/ver02.pth",
        "batch_size": 1,
        "num_workers": 16,
        "max_observations": 2,
        "pad_to_max_observations": 2,
        "min_scene_observations": None,
        "single_observation_policy": "model",
        "device": None,
        "tta": True,
        "tta_ops": None,
        "auto_masks": False,
        "no_submission_zip": True,
        "expected_size": (1080, 720),
        "no_size_check": True,
        "extra_data": "0",
        "description": "FUMO stage3 blending inference.",
    },
}

VALID_STAGES = ("stage1", "p_int", "stage2", "make_mask", "stage3")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return FINAL_DIR / path


def configured_paths() -> dict[str, Path | str]:
    paths = {key: resolve_path(value) for key, value in SETTINGS["paths"].items() if key != "split"}
    paths["split"] = SETTINGS["paths"]["split"]
    return paths


def iter_images(directory: Path, recursive: bool):
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def count_images(directory: Path | str, recursive: bool) -> int:
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(1 for _ in iter_images(directory, recursive))


def release_cuda_memory() -> None:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def timed_stage(name: str, image_count: int, func, timing: TimingSummary):
    start = time.perf_counter()
    try:
        output = func()
    except Exception:
        release_cuda_memory()
        raise
    elapsed = time.perf_counter() - start
    timing.add(name, image_count, elapsed)
    seconds_per_image = elapsed / max(image_count, 1)
    print(
        f"[timing] {name}: images={image_count} "
        f"elapsed={elapsed:.2f}s sec/image={seconds_per_image:.3f}s"
    )
    release_cuda_memory()
    return output


@contextmanager
def import_from(directory: Path, module_name: str):
    sys.path.insert(0, str(directory))
    previous = sys.modules.pop(module_name, None)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
        try:
            sys.path.remove(str(directory))
        except ValueError:
            pass


def selected_stages(cli_stage: str | None) -> list[str]:
    stages = [cli_stage] if cli_stage else list(SETTINGS["run_stages"])
    invalid = [stage for stage in stages if stage not in VALID_STAGES]
    if invalid:
        raise ValueError(f"Invalid stage(s): {invalid}. Valid stages: {VALID_STAGES}")
    return stages


def ensure_layout(paths: dict[str, Path | str]) -> None:
    for key, path in paths.items():
        if key != "split":
            Path(path).mkdir(parents=True, exist_ok=True)
    for path in (FINAL_DIR / "weights" / "stage1", FINAL_DIR / "weights" / "stage2", FINAL_DIR / "weights" / "stage3"):
        path.mkdir(parents=True, exist_ok=True)


def stage1_checkpoint() -> Path:
    return resolve_path(SETTINGS["stage1"]["checkpoint"])


def stage2_weight_path(key: str) -> Path:
    return resolve_path(SETTINGS["stage2"][key])


def stage3_checkpoint() -> Path:
    return resolve_path(SETTINGS["stage3"]["checkpoint"])


def run_stage1(paths: dict[str, Path | str], dry_run: bool) -> Path:
    cfg = SETTINGS["stage1"]
    print(f"[stage1] {paths['lq_dir']} -> {paths['is_dir']}")
    if dry_run:
        return Path(paths["is_dir"])
    with import_from(STAGE1_DIR, "inference") as module:
        return module.run_stage1(
            input_dir=paths["lq_dir"],
            output_dir=paths["is_dir"],
            checkpoint=stage1_checkpoint(),
            batch_size=cfg["batch_size"],
            resize_long_edge=cfg["resize_long_edge"],
            amp=cfg["amp"],
            tta=cfg["tta"],
            recursive=SETTINGS["recursive"],
            skip_existing=SETTINGS["skip_existing"],
            unsafe_load=cfg["unsafe_load"],
        )


def run_p_int(paths: dict[str, Path | str], dry_run: bool) -> Path:
    cfg = SETTINGS["p_int"]
    print(f"[p_int] {paths['is_dir']} -> {paths['p_int_dir']}")
    if dry_run:
        return Path(paths["p_int_dir"])
    with import_from(P_INT_CODE_DIR, "generate_p_int") as module:
        return module.run_p_int(
            input_dir=paths["is_dir"],
            output_dir=paths["p_int_dir"],
            model_path=cfg["model_path"],
            model_family=cfg["model_family"],
            recursive=SETTINGS["recursive"],
            overwrite=cfg["overwrite"],
            max_workers=cfg["max_workers"],
            max_long_side=cfg["max_long_side"],
        )


def run_stage2(paths: dict[str, Path | str], dry_run: bool) -> Path:
    cfg = SETTINGS["stage2"]
    print(f"[stage2] {paths['is_dir']} + {paths['p_int_dir']} -> {paths['is2_dir']}")
    if dry_run:
        return Path(paths["is2_dir"])
    with import_from(STAGE2_CODE_DIR, "inference") as module:
        return module.run_stage2(
            input_dir=paths["is_dir"],
            prior_dir=paths["p_int_dir"],
            output_dir=paths["is2_dir"],
            run_name=cfg["run_name"],
            pretrained_model_name_or_path=cfg["pretrained_model_name_or_path"],
            controlnet_dir=stage2_weight_path("controlnet_dir"),
            unet_dir=stage2_weight_path("unet_dir"),
            refine_net_path=stage2_weight_path("refine_net_path"),
            refine_head_path=stage2_weight_path("refine_head_path"),
            prompt=cfg["prompt"],
            resolution=cfg["resolution"],
            resolution_mode=cfg["resolution_mode"],
            resize=cfg["resize"],
            nafnet_width=cfg["nafnet_width"],
            nafnet_middle_blk_num=cfg["nafnet_middle_blk_num"],
            nafnet_enc_blk_nums=cfg["nafnet_enc_blk_nums"],
            nafnet_dec_blk_nums=cfg["nafnet_dec_blk_nums"],
            beta=cfg["beta"],
            residual_scale=cfg["residual_scale"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            recursive=SETTINGS["recursive"],
            save_prelim=cfg["save_prelim"],
            save_final=cfg["save_final"],
            save_input=cfg["save_input"],
            save_original_size=cfg["save_original_size"],
            skip_existing=SETTINGS["skip_existing"],
            tta=cfg["tta"],
            refine_tta=cfg["refine_tta"],
            limit=cfg["limit"],
            num_shards=cfg["num_shards"],
            shard_id=cfg["shard_id"],
        )


def run_make_mask(paths: dict[str, Path | str], dry_run: bool) -> Path:
    cfg = SETTINGS["make_mask"]
    print(
        f"[make_mask] {paths['lq_dir']} + {paths['is_dir']} -> {paths['drop_mask_dir']}; "
        f"{paths['is_dir']} + {paths['is2_dir']} -> {paths['reflection_mask_dir']}"
    )
    if dry_run:
        return Path(paths["drop_mask_dir"])
    with import_from(STAGE3_DIR, "inference") as module:
        module.generate_mask_dir(
            label="DROP_MASK",
            first_dir=Path(paths["lq_dir"]),
            second_dir=Path(paths["is_dir"]),
            mask_dir=Path(paths["drop_mask_dir"]),
            threshold_mode=module.resolve_mask_mode(cfg["threshold_mode"], Path(paths["drop_mask_dir"]).name),
            threshold=cfg["threshold"],
            threshold_percentile=cfg["threshold_percentile"],
            median_size=cfg["median_size"],
            overwrite=cfg["overwrite"],
            num_workers=cfg["num_workers"],
            chunksize=cfg["chunksize"],
        )
        module.generate_mask_dir(
            label="REFLECTION_MASK",
            first_dir=Path(paths["is_dir"]),
            second_dir=Path(paths["is2_dir"]),
            mask_dir=Path(paths["reflection_mask_dir"]),
            threshold_mode=module.resolve_mask_mode(cfg["threshold_mode"], Path(paths["reflection_mask_dir"]).name),
            threshold=cfg["threshold"],
            threshold_percentile=cfg["threshold_percentile"],
            median_size=cfg["median_size"],
            overwrite=cfg["overwrite"],
            num_workers=cfg["num_workers"],
            chunksize=cfg["chunksize"],
        )
    return Path(paths["drop_mask_dir"])


def run_stage3(paths: dict[str, Path | str], dry_run: bool) -> Path:
    cfg = SETTINGS["stage3"]
    print(
        f"[stage3] {paths['lq_dir']} + {paths['is_dir']} + {paths['is2_dir']} + "
        f"{paths['drop_mask_dir']} + {paths['reflection_mask_dir']} -> {paths['result_dir']}"
    )
    if dry_run:
        return Path(paths["result_dir"])
    with import_from(STAGE3_DIR, "inference") as module:
        return module.run_stage3(
            checkpoint=stage3_checkpoint(),
            data=paths["data_root"],
            split=paths["split"],
            final_dir=paths["result_dir"],
            save_run_artifacts=False,
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            max_observations=cfg["max_observations"],
            pad_to_max_observations=cfg["pad_to_max_observations"],
            min_scene_observations=cfg["min_scene_observations"],
            single_observation_policy=cfg["single_observation_policy"],
            device=cfg["device"],
            lq_dir=Path(paths["lq_dir"]).name,
            is_dir=Path(paths["is_dir"]).name,
            is2_dir=Path(paths["is2_dir"]).name,
            drop_mask_dir=Path(paths["drop_mask_dir"]).name,
            reflection_mask_dir=Path(paths["reflection_mask_dir"]).name,
            tta=cfg["tta"],
            tta_ops=cfg["tta_ops"],
            auto_masks=cfg["auto_masks"],
            no_submission_zip=cfg["no_submission_zip"],
            expected_size=cfg["expected_size"],
            no_size_check=cfg["no_size_check"],
            extra_data=cfg["extra_data"],
            description=cfg["description"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FUMO final inference pipeline.")
    parser.add_argument("--stage", choices=VALID_STAGES, default=None, help="Run only one stage.")
    parser.add_argument("--dry-run", action="store_true", help="Print the pipeline without running models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = selected_stages(args.stage)
    paths = configured_paths()
    ensure_layout(paths)

    if SETTINGS["cuda_visible_devices"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = SETTINGS["cuda_visible_devices"]

    print(f"Final dir: {FINAL_DIR}")
    print(f"Stages: {', '.join(stages)}")

    timing = TimingSummary()
    total_start = time.perf_counter()

    if "stage1" in stages:
        timed_stage(
            "stage1",
            count_images(paths["lq_dir"], SETTINGS["recursive"]),
            lambda: run_stage1(paths, args.dry_run),
            timing,
        )
    if "p_int" in stages:
        timed_stage(
            "p_int",
            count_images(paths["is_dir"], SETTINGS["recursive"]),
            lambda: run_p_int(paths, args.dry_run),
            timing,
        )
    if "stage2" in stages:
        timed_stage(
            "stage2",
            count_images(paths["is_dir"], SETTINGS["recursive"]),
            lambda: run_stage2(paths, args.dry_run),
            timing,
        )

    if "make_mask" in stages:
        timed_stage(
            "make_mask",
            count_images(paths["is2_dir"], SETTINGS["recursive"]),
            lambda: run_make_mask(paths, args.dry_run),
            timing,
        )

    if "stage3" in stages:
        timed_stage(
            "stage3",
            count_images(paths["is2_dir"], SETTINGS["recursive"]),
            lambda: run_stage3(paths, args.dry_run),
            timing,
        )

    timing.print_total(time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
