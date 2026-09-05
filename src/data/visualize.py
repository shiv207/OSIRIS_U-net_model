from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sar_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 2, 3) and image.shape[0] < image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        return np.repeat(image, 3, axis=-1)
    if image.shape[-1] == 2:
        z = np.zeros((*image.shape[:2], 1), dtype=image.dtype)
        return np.concatenate([image, z], axis=-1)
    return image[..., :3]


def overlay(image, mask, color=(1.0, 0.15, 0.1), alpha=0.45):
    base = np.clip(sar_display(image).astype(np.float32), 0, 1)
    if mask.ndim == 3:
        mask = mask.squeeze()
    out = base.copy()
    hit = mask > 0.5
    out[hit] = (1 - alpha) * base[hit] + alpha * np.array(color, np.float32)
    return np.clip(out, 0, 1)


def save_panel(dest: Path, sar, gt=None, pred=None, cv_pred=None, title=""):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    panels = [("SAR", sar_display(sar))]
    if gt is not None:
        panels += [("GT", gt), ("GT overlay", overlay(sar, gt))]
    if pred is not None:
        panels += [("pred", pred), ("pred overlay", overlay(sar, pred, (0.1, 0.85, 1)))]
    if cv_pred is not None:
        panels += [("cv", cv_pred), ("cv overlay", overlay(sar, cv_pred, (0.2, 1, 0.3)))]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, panels):
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(arr, 0, 1))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
