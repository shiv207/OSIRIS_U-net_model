from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.io import load_image, load_mask
from data.preprocess import binarize_mask, match_channels, normalize_sar, resize_pair
from utils import get_device, resolve_path


def read_split(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        img, msk = line.split("\t")
        rows.append((Path(img), Path(msk)))
    return rows


def _warp(image, mask, rng, rotate_deg, translate, scale):
    h, w = image.shape[:2]
    ang = float(rng.uniform(-rotate_deg, rotate_deg))
    tx = float(rng.uniform(-translate, translate) * w)
    ty = float(rng.uniform(-translate, translate) * h)
    sc = 1.0 + float(rng.uniform(-scale, scale))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if image.ndim == 2:
        image = image[..., None]
    return image, mask


def augment(image, mask, cfg, rng):
    aug = cfg.get("augmentation") or {}
    if not aug.get("enabled", True):
        return image, mask
    if rng.random() < float(aug.get("horizontal_flip", 0.5)):
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
    if rng.random() < float(aug.get("vertical_flip", 0.5)):
        image, mask = image[::-1].copy(), mask[::-1].copy()
    if aug.get("rot90", True):
        k = int(rng.integers(0, 4))
        image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
    if rng.random() < float(aug.get("affine_prob", 0)):
        image, mask = _warp(
            image,
            mask,
            rng,
            float(aug.get("affine_rotate_deg", 10)),
            float(aug.get("affine_translate", 0.05)),
            float(aug.get("affine_scale", 0.05)),
        )
    return image, mask


class SpillDataset(Dataset):
    def __init__(self, cfg, split: str, augment_on=None):
        self.cfg = cfg
        path = resolve_path(cfg, "splits_dir") / f"{split}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        self.pairs = read_split(path)
        self.size = int(cfg["dataset"]["image_size"])
        self.ch = int(cfg["dataset"]["in_channels"])
        self.pos = int(cfg["dataset"].get("mask_positive_min", 1))
        self.aug = (augment_on if augment_on is not None else True) and split == "train"
        self.rng = np.random.default_rng(int(cfg.get("seed", 42)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_p, msk_p = self.pairs[i]
        image = normalize_sar(load_image(img_p))
        mask = binarize_mask(load_mask(msk_p), self.pos)
        image, mask = resize_pair(image, mask, self.size)
        image = match_channels(image, self.ch)
        if self.aug:
            image, mask = augment(image, mask, self.cfg, self.rng)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float(),
            "image_path": str(img_p),
        }


def make_loader(cfg, split: str, shuffle=None):
    ds = SpillDataset(cfg, split, augment_on=(split == "train"))
    d = cfg["dataloader"]
    workers = int(d.get("num_workers", 0))
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=int(d["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(d.get("pin_memory", True)) and get_device().type == "cuda",
        persistent_workers=bool(d.get("persistent_workers", False)) and workers > 0,
    )
