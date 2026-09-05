from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_array(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.array(Image.open(path))
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def load_image(path: Path) -> np.ndarray:
    arr = load_array(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def load_mask(path: Path) -> np.ndarray:
    arr = load_array(path)
    return arr[..., 0] if arr.ndim == 3 else arr
