#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.blending import DirectFusionRestorationNet


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
MAP_EXTENSIONS = IMAGE_EXTENSIONS + (".npy",)
TTA_OPS_8 = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "hflip_rot90",
    "vflip_rot90",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone stage3 blending inference.")
    parser.add_argument("--checkpoint", default="weights", help=".pth file or a directory containing latest.pth/best.pth.")
    parser.add_argument("--data", default=None, help="Dataset root. Defaults to checkpoint config root or data/RDRF_dataset.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--final-dir", default=None, help="Optional directory for final per-image predictions.")
    parser.add_argument("--no-run-artifacts", dest="save_run_artifacts", action="store_false", default=True, help="Do not write timestamped run metadata/scene folders.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-observations", type=int, default=2)
    parser.add_argument("--pad-to-max-observations", type=int, default=2)
    parser.add_argument("--min-scene-observations", type=int, default=None)
    parser.add_argument("--single-observation-policy", choices=("model", "copy", "skip"), default="model")
    parser.add_argument("--device", default=None, help="cuda, cpu, or empty for config/default auto selection.")

    parser.add_argument("--lq-dir", default=None)
    parser.add_argument("--is-dir", default=None)
    parser.add_argument("--is1-dir", default=None)
    parser.add_argument("--is2-dir", default=None)
    parser.add_argument("--drop-mask-dir", default=None)
    parser.add_argument("--reflection-mask-dir", default=None)

    tta_group = parser.add_mutually_exclusive_group()
    tta_group.add_argument("--tta", dest="tta", action="store_true", default=True, help="Enable 8-direction TTA. Default: on.")
    tta_group.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA.")
    parser.add_argument("--tta-ops", nargs="+", default=None, help="Override TTA op list.")

    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--auto-masks", dest="auto_masks", action="store_true", default=True)
    mask_group.add_argument("--no-auto-masks", dest="auto_masks", action="store_false")
    parser.add_argument("--mask-threshold-mode", choices=("auto", "fixed", "percentile", "otsu"), default="auto")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--median-size", type=int, default=3)
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--mask-num-workers", type=int, default=None, help="Defaults to --num-workers. Use 0 for all CPU cores.")
    parser.add_argument("--mask-chunksize", type=int, default=8)

    parser.add_argument("--submission-zip", default=None, help="Default: <run_dir>/submission.zip")
    parser.add_argument("--no-submission-zip", action="store_true")
    parser.add_argument("--expected-size", nargs=2, type=int, default=(1080, 720), metavar=("W", "H"))
    parser.add_argument("--no-size-check", action="store_true")
    parser.add_argument("--extra-data", choices=("0", "1"), default="0")
    parser.add_argument(
        "--description",
        default="Direct multi-observation stage3 blending restoration. No extra training data was used.",
    )
    return parser.parse_args()


class ObservationBlendingSceneDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="test",
        gt_dir="GT",
        is2_dir="IS2",
        is1_dir="IS",
        include_is1=False,
        lq_dir="LQ",
        include_lq=False,
        drop_mask_dir="DROP_MASK",
        reflection_mask_dir="REFLECTION_MASK",
        max_observations=2,
        min_scene_observations=None,
        has_gt=False,
        **_unused,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_observations = None if max_observations is None else int(max_observations)
        self.min_scene_observations = None if min_scene_observations is None else int(min_scene_observations)
        self.has_gt = bool(has_gt)
        self.include_is1 = bool(include_is1)
        self.include_lq = bool(include_lq)

        if self.max_observations is not None and self.max_observations < 1:
            raise ValueError("--max-observations must be >= 1")
        if self.min_scene_observations is not None and self.min_scene_observations < 1:
            raise ValueError("--min-scene-observations must be >= 1")

        gt_paths = {}
        if self.has_gt:
            gt_paths = list_paths_by_stem(resolve_data_dir(self.root_dir, self.split, gt_dir))

        is2_paths = list_paths_by_stem(resolve_data_dir(self.root_dir, self.split, is2_dir))
        is1_paths = list_paths_by_stem(resolve_data_dir(self.root_dir, self.split, is1_dir)) if self.include_is1 else {}
        lq_paths = list_paths_by_stem(resolve_data_dir(self.root_dir, self.split, lq_dir)) if self.include_lq else {}
        drop_paths = list_paths_by_stem(resolve_data_dir(self.root_dir, self.split, drop_mask_dir), MAP_EXTENSIONS)
        refl_paths = list_paths_by_stem(resolve_data_dir(self.root_dir, self.split, reflection_mask_dir), MAP_EXTENSIONS)

        scenes = defaultdict(list)
        missing = []
        for stem, is2_path in sorted(is2_paths.items()):
            scene_id = scene_id_from_stem(stem)
            item = {
                "stem": stem,
                "name": f"{stem}.png",
                "is2_path": is2_path,
                "is1_path": is1_paths.get(stem) if self.include_is1 else None,
                "lq_path": lq_paths.get(stem) if self.include_lq else None,
                "drop_path": drop_paths.get(stem),
                "refl_path": refl_paths.get(stem),
                "gt_path": gt_paths.get(scene_id) if self.has_gt else None,
            }
            if (
                item["drop_path"] is None
                or item["refl_path"] is None
                or (self.include_is1 and item["is1_path"] is None)
                or (self.include_lq and item["lq_path"] is None)
                or (self.has_gt and item["gt_path"] is None)
            ):
                missing.append(stem)
                continue
            scenes[scene_id].append(item)

        if self.min_scene_observations is not None:
            scenes = {
                scene_id: items
                for scene_id, items in scenes.items()
                if len(items) >= self.min_scene_observations
            }
        if not scenes:
            raise ValueError(
                f"No matched scenes in root={self.root_dir}, split={self.split}. "
                "Expected IS2, DROP_MASK, and REFLECTION_MASK for each observation."
            )

        self.scenes = dict(sorted(scenes.items()))
        self.scene_ids = list(self.scenes.keys())
        self.missing_stems = missing

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        scene_id = self.scene_ids[index]
        all_items = list(self.scenes[scene_id])
        items = all_items[: self.max_observations] if self.max_observations is not None else all_items

        is2 = [load_rgb_tensor(item["is2_path"]) for item in items]
        is1 = [load_rgb_tensor(item["is1_path"]) for item in items] if self.include_is1 else None
        lq = [load_rgb_tensor(item["lq_path"]) for item in items] if self.include_lq else None
        drop = [load_mask_tensor(item["drop_path"]) for item in items]
        reflection = [load_mask_tensor(item["refl_path"]) for item in items]
        validate_sizes(scene_id, is2, drop, reflection, is1=is1, lq=lq)

        sample = {
            "scene_id": scene_id,
            "stems": [item["stem"] for item in items],
            "names": [item["name"] for item in items],
            "target_stems": [item["stem"] for item in all_items],
            "target_names": [item["name"] for item in all_items],
            "is2": torch.stack(is2, dim=0),
            "drop": torch.stack(drop, dim=0),
            "reflection": torch.stack(reflection, dim=0),
        }
        if is1 is not None:
            sample["is1"] = torch.stack(is1, dim=0)
        if lq is not None:
            sample["lq"] = torch.stack(lq, dim=0)
        return sample


def blending_collate_fn(batch, max_observations=2):
    if not batch:
        raise ValueError("Empty batch")
    if max_observations is None:
        max_observations = max(sample["is2"].size(0) for sample in batch)
    max_observations = int(max_observations)
    batch_size = len(batch)
    max_h = max(sample["is2"].shape[-2] for sample in batch)
    max_w = max(sample["is2"].shape[-1] for sample in batch)

    output = {
        "is2": torch.zeros(batch_size, max_observations, 3, max_h, max_w),
        "drop": torch.zeros(batch_size, max_observations, 1, max_h, max_w),
        "reflection": torch.zeros(batch_size, max_observations, 1, max_h, max_w),
        "valid": torch.zeros(batch_size, max_observations, dtype=torch.bool),
        "original_size": [],
        "scene_id": [],
        "stems": [],
        "names": [],
        "target_stems": [],
        "target_names": [],
    }
    has_is1 = "is1" in batch[0]
    has_lq = "lq" in batch[0]
    if has_is1:
        output["is1"] = torch.zeros(batch_size, max_observations, 3, max_h, max_w)
    if has_lq:
        output["lq"] = torch.zeros(batch_size, max_observations, 3, max_h, max_w)

    for batch_index, sample in enumerate(batch):
        n = min(sample["is2"].size(0), max_observations)
        h, w = sample["is2"].shape[-2:]
        output["is2"][batch_index, :n, :, :h, :w] = sample["is2"][:n]
        output["drop"][batch_index, :n, :, :h, :w] = sample["drop"][:n]
        output["reflection"][batch_index, :n, :, :h, :w] = sample["reflection"][:n]
        if has_is1:
            output["is1"][batch_index, :n, :, :h, :w] = sample["is1"][:n]
        if has_lq:
            output["lq"][batch_index, :n, :, :h, :w] = sample["lq"][:n]
        output["valid"][batch_index, :n] = True
        output["original_size"].append((h, w))
        output["scene_id"].append(sample["scene_id"])
        output["stems"].append(sample["stems"][:n])
        output["names"].append(sample["names"][:n])
        output["target_stems"].append(sample["target_stems"])
        output["target_names"].append(sample["target_names"])
    return output


def run_stage3_from_args(args):
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    config = normalize_config(checkpoint.get("config") if isinstance(checkpoint, dict) else None)
    dataset_cfg = resolve_dataset_config(config, args)

    if args.auto_masks:
        ensure_masks(dataset_cfg, args)

    device = resolve_device(args.device, config)
    model = build_model(config).to(device)
    iteration = load_model_state(model, checkpoint)
    ema_used = apply_checkpoint_ema_to_model(checkpoint, model)
    model.eval()

    dataset = ObservationBlendingSceneDataset(**dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: blending_collate_fn(batch, max_observations=args.pad_to_max_observations),
    )

    save_run_artifacts = bool(getattr(args, "save_run_artifacts", True))
    run_dir = create_run_dir(args.output_root, f"{Path(dataset_cfg['root_dir']).name}_{args.split}") if save_run_artifacts else None
    final_dir = Path(args.final_dir) if getattr(args, "final_dir", None) else None
    if final_dir is not None:
        final_dir.mkdir(parents=True, exist_ok=True)
    if run_dir is not None:
        write_json(config, run_dir / "checkpoint_config.json")

    tta_ops = tuple(args.tta_ops or get_tta_ops_from_config(config, "inference"))
    saved_paths = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="stage3 inference"):
            batch = move_batch_to_device(batch, device)
            batch, outputs = apply_single_observation_policy(
                model,
                batch,
                policy=args.single_observation_policy,
                tta_enabled=args.tta,
                tta_ops=tta_ops,
            )
            if batch is None:
                continue
            saved_paths.extend(save_predictions(outputs["pred"].cpu(), batch, run_dir, final_dir=final_dir))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    runtime_per_image = elapsed / max(len(saved_paths), 1)

    submission_zip = None
    if not args.no_submission_zip:
        if args.submission_zip:
            submission_zip = Path(args.submission_zip)
        elif run_dir is not None:
            submission_zip = run_dir / "submission.zip"
        elif final_dir is not None:
            submission_zip = final_dir / "submission.zip"
        else:
            raise ValueError("submission_zip must be set when run artifacts and final_dir are disabled.")
        readme_text = make_submission_readme(runtime_per_image, device, args.extra_data, args.description)
        expected_size = None if args.no_size_check else tuple(args.expected_size)
        create_submission_zip(saved_paths, submission_zip, readme_text, expected_size=expected_size)

    summary = {
        "checkpoint": str(checkpoint_path),
        "iteration": iteration,
        "ema_used": bool(ema_used),
        "data": str(dataset_cfg["root_dir"]),
        "split": args.split,
        "num_scenes": len(dataset),
        "num_saved_images": len(saved_paths),
        "max_observations": args.max_observations,
        "pad_to_max_observations": args.pad_to_max_observations,
        "tta_enabled": bool(args.tta),
        "tta_ops": list(tta_ops) if args.tta else None,
        "runtime_seconds": elapsed,
        "runtime_per_image_seconds": runtime_per_image,
        "submission_zip": str(submission_zip) if submission_zip else None,
        "final_dir": str(final_dir) if final_dir is not None else None,
    }
    if run_dir is not None:
        write_json(summary, run_dir / "inference.json")
    print(
        "Inference complete: "
        f"run_dir={run_dir if run_dir is not None else 'disabled'} "
        f"scenes={summary['num_scenes']} "
        f"saved_images={summary['num_saved_images']} "
        f"tta={summary['tta_enabled']} "
        f"runtime_per_image={runtime_per_image:.4f}s "
        f"submission_zip={submission_zip}"
    )
    return final_dir if final_dir is not None else run_dir


def resolve_checkpoint_path(path):
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    candidates = [path / "latest.pth", path / "best.pth", path / "model.pth"]
    candidates.extend(sorted(path.glob("*.pth")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .pth checkpoint found in: {path}")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_config(config):
    config = dict(config or {})
    config.setdefault("runtime", {})
    config.setdefault("model", {})
    config["model"].setdefault("arch", "direct_fusion_restoration")
    config["model"].setdefault("args", {})
    config.setdefault("dataset", {})
    return config


def build_model(config):
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch") or model_cfg.get("name") or "direct_fusion_restoration"
    if arch != "direct_fusion_restoration":
        raise ValueError(f"Unsupported model arch for stage3 inference: {arch}")
    return DirectFusionRestorationNet(**model_cfg.get("args", {}))


def load_model_state(model, checkpoint):
    if isinstance(checkpoint, dict):
        state = first_present(checkpoint, ("model", "model_state_dict", "state_dict", "params", "params_ema"))
        iteration = int(checkpoint.get("iteration", checkpoint.get("iter", checkpoint.get("global_step", 0))))
    else:
        state = checkpoint
        iteration = 0
    if state is None:
        state = checkpoint
    load_state_dict_with_module_fallback(model, state)
    return iteration


def first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def load_state_dict_with_module_fallback(model, state):
    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        pass
    stripped = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(stripped)


def apply_checkpoint_ema_to_model(checkpoint, model):
    if not isinstance(checkpoint, dict):
        return False
    ema_state = checkpoint.get("ema")
    if isinstance(ema_state, dict) and "shadow" in ema_state:
        ema_state = ema_state["shadow"]
    elif not isinstance(ema_state, dict):
        ema_state = checkpoint.get("params_ema")
    if not isinstance(ema_state, dict):
        return False

    model_state = model.state_dict()
    load_state = {}
    for name, value in ema_state.items():
        target_name = match_state_name(name, model_state)
        if target_name is None:
            continue
        load_state[target_name] = value.to(device=model_state[target_name].device, dtype=model_state[target_name].dtype)
    if not load_state:
        return False
    model.load_state_dict(load_state, strict=False)
    return True


def match_state_name(name, state):
    if name in state:
        return name
    if name.startswith("module.") and name.removeprefix("module.") in state:
        return name.removeprefix("module.")
    prefixed = f"module.{name}"
    return prefixed if prefixed in state else None


def resolve_dataset_config(config, args):
    dataset_root = config.get("dataset", {})
    split_cfg = dict(dataset_root.get(args.split, dataset_root.get("test", dataset_root.get("validation", {}))))
    model_args = config.get("model", {}).get("args", {})

    root_dir = args.data or split_cfg.get("root") or "data/RDRF_dataset"
    cfg = {
        "root_dir": root_dir,
        "split": args.split,
        "gt_dir": split_cfg.get("gt_dir", "GT"),
        "is2_dir": args.is2_dir or split_cfg.get("is2_dir", "IS2"),
        "is1_dir": args.is1_dir or args.is_dir or split_cfg.get("is1_dir", split_cfg.get("is_dir", "IS")),
        "include_is1": bool(split_cfg.get("include_is1", False) or model_args.get("use_is1_input", False)),
        "lq_dir": args.lq_dir or split_cfg.get("lq_dir", "LQ"),
        "include_lq": bool(split_cfg.get("include_lq", False) or model_args.get("use_lq_input", False)),
        "drop_mask_dir": args.drop_mask_dir or split_cfg.get("drop_mask_dir", "DROP_MASK"),
        "reflection_mask_dir": args.reflection_mask_dir or split_cfg.get("reflection_mask_dir", "REFLECTION_MASK"),
        "max_observations": args.max_observations,
        "min_scene_observations": args.min_scene_observations,
        "has_gt": False,
    }
    return cfg


def resolve_device(device_arg, config):
    requested = device_arg or config.get("runtime", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested if requested else "cuda" if torch.cuda.is_available() else "cpu")


def get_tta_ops_from_config(config, section):
    section_cfg = config.get(section, {})
    tta_cfg = section_cfg.get("tta", {})
    if isinstance(tta_cfg, dict):
        return tuple(tta_cfg.get("ops", TTA_OPS_8))
    return TTA_OPS_8


def scene_id_from_stem(stem):
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    if "-" in stem:
        return stem.split("-", 1)[0]
    return stem


def resolve_data_dir(root_dir, split, value):
    path = Path(value)
    if path.is_absolute():
        return path
    split_candidate = Path(root_dir) / split / path
    if split_candidate.exists():
        return split_candidate
    root_candidate = Path(root_dir) / path
    if root_candidate.exists():
        return root_candidate
    return split_candidate


def list_paths_by_stem(directory, extensions=IMAGE_EXTENSIONS):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    paths = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate stem in {directory}: {path.stem}")
        paths[path.stem] = path
    if not paths:
        raise ValueError(f"No matching files found in: {directory}")
    return paths


def load_rgb_tensor(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask_tensor(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = torch.from_numpy(np.load(path)).float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)
        return mask.clamp(0.0, 1.0).contiguous()
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
    return torch.from_numpy(array).clamp(0.0, 1.0).contiguous()


def validate_sizes(scene_id, is2, drop, reflection, is1=None, lq=None):
    expected = is2[0].shape[-2:]
    groups = [("IS2", is2), ("DROP_MASK", drop), ("REFLECTION_MASK", reflection)]
    if is1 is not None:
        groups.append(("IS", is1))
    if lq is not None:
        groups.append(("LQ", lq))
    for group_name, tensors in groups:
        for tensor in tensors:
            if tensor.shape[-2:] != expected:
                raise ValueError(f"Size mismatch in scene {scene_id}: {group_name} has {tensor.shape[-2:]}, expected {expected}")


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def model_forward(model, batch):
    return model(
        batch["is2"],
        batch["drop"],
        batch["reflection"],
        batch["valid"],
        is1=batch.get("is1"),
        lq=batch.get("lq"),
    )


def model_forward_with_tta(model, batch, enabled=True, ops=TTA_OPS_8):
    if not enabled:
        return model_forward(model, batch)
    collected = {}
    for op in ops:
        augmented = apply_batch_tta_op(batch, op)
        outputs = model_forward(model, augmented)
        for name, value in outputs.items():
            if torch.is_tensor(value):
                restored = invert_tta_op(value, op) if value.ndim >= 2 else value
                collected.setdefault(name, []).append(restored)
    averaged = {}
    for name, values in collected.items():
        value = torch.stack(values, dim=0).mean(dim=0)
        averaged[name] = value.clamp(0.0, 1.0) if name in {"pred", "anchor"} else value
    return averaged


def apply_batch_tta_op(batch, op):
    augmented = dict(batch)
    for key in ("is2", "is1", "lq", "drop", "reflection"):
        value = augmented.get(key)
        if torch.is_tensor(value):
            augmented[key] = apply_tta_op(value, op)
    return augmented


def apply_tta_op(tensor, op):
    if op == "identity":
        return tensor
    if op == "rot90":
        return torch.rot90(tensor, 1, (-2, -1))
    if op == "rot180":
        return torch.rot90(tensor, 2, (-2, -1))
    if op == "rot270":
        return torch.rot90(tensor, 3, (-2, -1))
    if op == "hflip":
        return torch.flip(tensor, dims=(-1,))
    if op == "vflip":
        return torch.flip(tensor, dims=(-2,))
    if op == "hflip_rot90":
        return torch.rot90(torch.flip(tensor, dims=(-1,)), 1, (-2, -1))
    if op == "vflip_rot90":
        return torch.rot90(torch.flip(tensor, dims=(-2,)), 1, (-2, -1))
    raise ValueError(f"Unsupported TTA op: {op}")


def invert_tta_op(tensor, op):
    if op == "identity":
        return tensor
    if op == "rot90":
        return torch.rot90(tensor, 3, (-2, -1))
    if op == "rot180":
        return torch.rot90(tensor, 2, (-2, -1))
    if op == "rot270":
        return torch.rot90(tensor, 1, (-2, -1))
    if op == "hflip":
        return torch.flip(tensor, dims=(-1,))
    if op == "vflip":
        return torch.flip(tensor, dims=(-2,))
    if op == "hflip_rot90":
        return torch.flip(torch.rot90(tensor, 3, (-2, -1)), dims=(-1,))
    if op == "vflip_rot90":
        return torch.flip(torch.rot90(tensor, 3, (-2, -1)), dims=(-2,))
    raise ValueError(f"Unsupported TTA op: {op}")


def apply_single_observation_policy(model, batch, policy="model", min_model_observations=2, tta_enabled=True, tta_ops=TTA_OPS_8):
    policy = str(policy or "model").lower()
    if policy == "model":
        return batch, model_forward_with_tta(model, batch, enabled=tta_enabled, ops=tta_ops)

    counts = [max(1, int(batch["valid"][idx].sum().item())) for idx in range(batch["is2"].size(0))]
    needs_model = [count >= min_model_observations for count in counts]
    if policy == "skip":
        keep = [idx for idx, use_model in enumerate(needs_model) if use_model]
        if not keep:
            return None, None
        batch = filter_batch(batch, keep)
        return batch, model_forward_with_tta(model, batch, enabled=tta_enabled, ops=tta_ops)
    if policy != "copy":
        raise ValueError(f"Unsupported single_observation_policy: {policy}")

    if not any(needs_model):
        pred = batch["is2"][:, 0].clone()
        return batch, {"pred": pred, "anchor": pred.clone(), "residual": torch.zeros_like(pred)}
    outputs = model_forward_with_tta(model, batch, enabled=tta_enabled, ops=tta_ops)
    pred = outputs["pred"].clone()
    for idx, use_model in enumerate(needs_model):
        if not use_model:
            pred[idx] = batch["is2"][idx, 0]
    outputs["pred"] = pred
    return batch, outputs


def filter_batch(batch, indices):
    index_tensor = torch.as_tensor(indices, device=batch["is2"].device, dtype=torch.long)
    batch_size = batch["is2"].size(0)
    filtered = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.size(0) == batch_size:
            filtered[key] = value.index_select(0, index_tensor)
        elif isinstance(value, list) and len(value) == batch_size:
            filtered[key] = [value[index] for index in indices]
        else:
            filtered[key] = value
    return filtered


def save_predictions(pred, batch, run_dir=None, final_dir=None):
    run_dir = Path(run_dir) if run_dir is not None else None
    if final_dir is None and run_dir is None:
        raise ValueError("Either final_dir or run_dir must be provided for saving predictions.")
    output_dir = Path(final_dir) if final_dir is not None else run_dir / "pred"
    scene_dir = run_dir / "scene" if run_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if scene_dir is not None:
        scene_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for batch_index, stems in enumerate(batch["target_stems"]):
        height, width = batch["original_size"][batch_index]
        image = tensor_to_image(pred[batch_index, :, :height, :width])
        if scene_dir is not None:
            image.save(scene_dir / f"{batch['scene_id'][batch_index]}.png")
        for stem in stems:
            save_path = output_dir / f"{stem}.png"
            image.save(save_path)
            saved.append(save_path)
    return saved


def tensor_to_image(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    array = (tensor[:3].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def ensure_masks(dataset_cfg, args):
    root = Path(dataset_cfg["root_dir"])
    split = dataset_cfg["split"]
    split_root = root / split
    if not split_root.exists():
        return
    mask_workers = args.num_workers if args.mask_num_workers is None else args.mask_num_workers
    specs = [
        ("DROP_MASK", dataset_cfg["lq_dir"], dataset_cfg["is1_dir"], dataset_cfg["drop_mask_dir"]),
        ("REFLECTION_MASK", dataset_cfg["is1_dir"], dataset_cfg["is2_dir"], dataset_cfg["reflection_mask_dir"]),
    ]
    for label, first_dir_name, second_dir_name, mask_dir_name in specs:
        first_dir = resolve_data_dir(root, split, first_dir_name)
        second_dir = resolve_data_dir(root, split, second_dir_name)
        mask_dir = resolve_data_dir(root, split, mask_dir_name)
        if not first_dir.is_dir() or not second_dir.is_dir():
            continue
        generate_mask_dir(
            label=label,
            first_dir=first_dir,
            second_dir=second_dir,
            mask_dir=mask_dir,
            threshold_mode=resolve_mask_mode(args.mask_threshold_mode, mask_dir_name),
            threshold=args.threshold,
            threshold_percentile=args.threshold_percentile,
            median_size=args.median_size,
            overwrite=args.overwrite_masks,
            num_workers=mask_workers,
            chunksize=args.mask_chunksize,
        )


def resolve_mask_mode(mode, mask_dir_name):
    if mode != "auto":
        return mode
    return "otsu" if "OTSU" in str(mask_dir_name).upper() else "fixed"


def generate_mask_dir(label, first_dir, second_dir, mask_dir, threshold_mode, threshold, threshold_percentile, median_size, overwrite, num_workers, chunksize):
    first_paths = collect_paths_by_stem(first_dir, IMAGE_EXTENSIONS)
    second_paths = collect_paths_by_stem(second_dir, IMAGE_EXTENSIONS)
    mask_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for stem, first_path in first_paths.items():
        second_path = second_paths.get(stem)
        if second_path is None:
            continue
        mask_path = mask_dir / f"{stem}.png"
        if mask_path.exists() and not overwrite:
            continue
        tasks.append((stem, first_path, second_path, mask_path, threshold_mode, threshold, threshold_percentile, median_size))
    if not tasks:
        return
    worker_count = os.cpu_count() if int(num_workers) == 0 else int(num_workers)
    worker_count = max(1, worker_count or 1)
    if worker_count == 1:
        written = [process_mask_task(task) for task in tqdm(tasks, desc=f"auto {label}")]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            written = list(tqdm(executor.map(process_mask_task, tasks, chunksize=chunksize), total=len(tasks), desc=f"auto {label} x{worker_count}"))
    print(f"Auto masks {label}: written={sum(written)} output={mask_dir}")


def collect_paths_by_stem(directory, extensions):
    paths = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            paths[path.stem] = path
    return paths


def process_mask_task(task):
    _stem, first_path, second_path, mask_path, threshold_mode, threshold, threshold_percentile, median_size = task
    first = Image.open(first_path).convert("RGB")
    second = Image.open(second_path).convert("RGB")
    if first.size != second.size:
        raise ValueError(f"Size mismatch while making mask: {first_path} vs {second_path}")
    score = compute_positive_luma_diff(first, second)
    mask = make_binary_mask(score, threshold_mode, threshold, threshold_percentile, median_size)
    mask.save(mask_path)
    return 1


def compute_positive_luma_diff(first_image, second_image):
    first = np.asarray(first_image, dtype=np.float32)
    second = np.asarray(second_image, dtype=np.float32)
    diff = np.maximum(first - second, 0.0)
    return np.clip(0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2], 0.0, 255.0)


def make_binary_mask(score, threshold_mode, threshold, threshold_percentile, median_size):
    if threshold_mode == "fixed":
        threshold_value = normalize_threshold(threshold)
    elif threshold_mode == "percentile":
        threshold_value = float(np.percentile(score, threshold_percentile))
    elif threshold_mode == "otsu":
        threshold_value = otsu_threshold(score)
    else:
        raise ValueError(f"Unsupported mask threshold mode: {threshold_mode}")
    if threshold_mode == "otsu" and threshold_value <= 0.0:
        mask = np.zeros_like(score, dtype=np.uint8)
    else:
        mask = (score > threshold_value).astype(np.uint8) * 255
    image = Image.fromarray(mask)
    if median_size and int(median_size) > 1:
        if int(median_size) % 2 == 0:
            raise ValueError("--median-size must be odd")
        image = image.filter(ImageFilter.MedianFilter(size=int(median_size)))
    return image


def normalize_threshold(threshold):
    threshold = float(threshold)
    if threshold <= 1.0:
        threshold *= 255.0
    return float(np.clip(threshold, 0.0, 255.0))


def otsu_threshold(score):
    values = np.clip(score.round(), 0, 255).astype(np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    bins = np.arange(256, dtype=np.float64)
    sum_total = float((bins * hist).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best = -math.inf
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
        if between > best:
            best = between
            threshold = float(index)
    return threshold


def create_run_dir(root, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / slugify(name) / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = Path(root) / slugify(name) / f"{timestamp}_{suffix:03d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def slugify(value):
    keep = []
    for char in str(value).strip():
        keep.append(char if char.isalnum() or char in "._-" else "_")
    slug = "".join(keep).strip("._-")
    return slug or "run"


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True, default=str)


def make_submission_readme(runtime_per_image, device, extra_data, description):
    return (
        f"runtime per image [s] : {runtime_per_image:.4f}\n"
        f"CPU[1] / GPU[0] : {'1' if device.type == 'cpu' else '0'}\n"
        f"Extra Data [1] / No Extra Data [0] : {extra_data}\n"
        f"Other description: {description}\n"
    )


def create_submission_zip(pred_paths, output_path, readme_text, expected_size=(1080, 720)):
    if not pred_paths:
        raise ValueError("No prediction images were generated.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [Path(path).name for path in pred_paths]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate prediction names cannot be submitted: {duplicates[:20]}")
    if expected_size is not None:
        bad_sizes = []
        for path in pred_paths:
            with Image.open(path) as image:
                if image.size != tuple(expected_size):
                    bad_sizes.append(f"{Path(path).name}: {image.size[0]}x{image.size[1]}")
        if bad_sizes:
            preview = "\n".join(bad_sizes[:20])
            raise ValueError(f"Submission images must be {expected_size[0]}x{expected_size[1]}:\n{preview}")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(pred_paths, key=lambda item: Path(item).name):
            archive.write(path, arcname=Path(path).name)
        archive.writestr("readme.txt", readme_text)



def run_stage3(
    checkpoint="weights/stage3",
    data="data",
    split="test",
    output_root="data/test/stage3_runs",
    final_dir=None,
    save_run_artifacts=True,
    batch_size=1,
    num_workers=16,
    max_observations=2,
    pad_to_max_observations=2,
    min_scene_observations=None,
    single_observation_policy="model",
    device=None,
    lq_dir="LQ",
    is_dir="IS",
    is1_dir=None,
    is2_dir="IS2",
    drop_mask_dir="drop_mask",
    reflection_mask_dir="reflection_mask",
    tta=True,
    tta_ops=None,
    auto_masks=True,
    mask_threshold_mode="auto",
    threshold=20.0,
    threshold_percentile=95.0,
    median_size=3,
    overwrite_masks=False,
    mask_num_workers=None,
    mask_chunksize=8,
    submission_zip=None,
    no_submission_zip=True,
    expected_size=(1080, 720),
    no_size_check=True,
    extra_data="0",
    description="FUMO stage3 blending inference.",
):
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        data=str(data) if data is not None else None,
        split=split,
        output_root=str(output_root),
        final_dir=str(final_dir) if final_dir is not None else None,
        save_run_artifacts=save_run_artifacts,
        batch_size=batch_size,
        num_workers=num_workers,
        max_observations=max_observations,
        pad_to_max_observations=pad_to_max_observations,
        min_scene_observations=min_scene_observations,
        single_observation_policy=single_observation_policy,
        device=device,
        lq_dir=lq_dir,
        is_dir=is_dir,
        is1_dir=is1_dir,
        is2_dir=is2_dir,
        drop_mask_dir=drop_mask_dir,
        reflection_mask_dir=reflection_mask_dir,
        tta=tta,
        tta_ops=tta_ops,
        auto_masks=auto_masks,
        mask_threshold_mode=mask_threshold_mode,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        median_size=median_size,
        overwrite_masks=overwrite_masks,
        mask_num_workers=mask_num_workers,
        mask_chunksize=mask_chunksize,
        submission_zip=submission_zip,
        no_submission_zip=no_submission_zip,
        expected_size=expected_size,
        no_size_check=no_size_check,
        extra_data=extra_data,
        description=description,
    )
    return run_stage3_from_args(args)


def main():
    return run_stage3_from_args(parse_args())


if __name__ == "__main__":
    main()
