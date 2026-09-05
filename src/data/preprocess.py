from __future__ import annotations

import cv2
import numpy as np


def normalize_sar(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    out = np.empty_like(image)
    for c in range(image.shape[-1]):
        ch = image[..., c]
        ok = np.isfinite(ch)
        if not ok.any():
            out[..., c] = 0
            continue
        vals = ch[ok]
        lo, hi = np.percentile(vals, (1, 99))
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            out[..., c] = 0
            continue
        scaled = np.clip((ch - lo) / (hi - lo), 0, 1)
        scaled[~ok] = 0
        out[..., c] = scaled
    return out


def binarize_mask(mask: np.ndarray, positive_min: int = 1) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask >= positive_min).astype(np.uint8)


def resize_pair(image, mask, size: int):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 2:
        image = image[..., None]
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def match_channels(image: np.ndarray, n: int) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    c = image.shape[-1]
    if c == n:
        return image
    if c == 1:
        return np.repeat(image, n, axis=-1)
    if c > n:
        return image[..., :n]
    pad = np.zeros((*image.shape[:2], n - c), dtype=image.dtype)
    return np.concatenate([image, pad], axis=-1)
