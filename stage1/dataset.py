# -*- coding: utf-8 -*-
import random
from pathlib import Path
from random import randrange
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def prefix_of(path: Path) -> str:
    return path.stem.split("-", 1)[0]


class ReflectionPairDataset(data.Dataset):
    def __init__(
        self,
        data_root: str | Path,
        crop_size: int | Tuple[int, int] = 256,
        split: str = "train",
        lq_dir_name: str = "LQ",
        gt_dir_name: str = "LQ_ReflectionOnly",
        random_flip: bool = True,
        random_rotate: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.lq_dir = self.data_root / split / lq_dir_name
        self.gt_dir = self.data_root / split / gt_dir_name
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

        if not self.lq_dir.is_dir():
            raise FileNotFoundError(f"LQ directory not found: {self.lq_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        gt_by_prefix = self._collect_gt_by_prefix(self.gt_dir)
        self.pairs = self._collect_pairs(self.lq_dir, gt_by_prefix)
        if not self.pairs:
            raise RuntimeError(f"no train pairs found from {self.lq_dir} and {self.gt_dir}")

    @staticmethod
    def _image_files(directory: Path) -> List[Path]:
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)

    def _collect_gt_by_prefix(self, gt_dir: Path) -> Dict[str, Path]:
        gt_by_prefix: Dict[str, Path] = {}
        for path in self._image_files(gt_dir):
            prefix = prefix_of(path)
            if prefix in gt_by_prefix:
                raise RuntimeError(
                    f"multiple GT images for prefix '{prefix}': {gt_by_prefix[prefix].name}, {path.name}"
                )
            gt_by_prefix[prefix] = path
        return gt_by_prefix

    def _collect_pairs(self, lq_dir: Path, gt_by_prefix: Dict[str, Path]) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        missing = set()
        for lq_path in self._image_files(lq_dir):
            prefix = prefix_of(lq_path)
            gt_path = gt_by_prefix.get(prefix)
            if gt_path is None:
                missing.add(prefix)
                continue
            pairs.append((lq_path, gt_path))
        if missing:
            print(f"Skipped {len(missing)} LQ prefixes without GT: {sorted(missing)[:10]}")
        return pairs

    @staticmethod
    def _resize_for_crop(input_img: Image.Image, gt_img: Image.Image, crop_width: int, crop_height: int):
        if gt_img.size != input_img.size:
            gt_img = gt_img.resize(input_img.size, Image.BICUBIC)

        width, height = input_img.size
        new_width = max(width, crop_width)
        new_height = max(height, crop_height)
        if (new_width, new_height) != input_img.size:
            input_img = input_img.resize((new_width, new_height), Image.LANCZOS)
            gt_img = gt_img.resize((new_width, new_height), Image.LANCZOS)
        return input_img, gt_img

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        lq_path, gt_path = self.pairs[index]

        input_img = Image.open(lq_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        input_img, gt_img = self._resize_for_crop(input_img, gt_img, crop_width, crop_height)

        width, height = input_img.size
        x = randrange(0, width - crop_width + 1)
        y = randrange(0, height - crop_height + 1)
        input_crop = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop = gt_img.crop((x, y, x + crop_width, y + crop_height))

        input_tensor = TF.normalize(TF.to_tensor(input_crop), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        gt_tensor = TF.to_tensor(gt_crop)

        if self.random_flip and random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=[-1])
            gt_tensor = torch.flip(gt_tensor, dims=[-1])

        if self.random_rotate:
            rotate_k = random.randrange(4)
            if rotate_k:
                input_tensor = torch.rot90(input_tensor, k=rotate_k, dims=(1, 2))
                gt_tensor = torch.rot90(gt_tensor, k=rotate_k, dims=(1, 2))

        return input_tensor, gt_tensor, lq_path.name

    def __len__(self):
        return len(self.pairs)
