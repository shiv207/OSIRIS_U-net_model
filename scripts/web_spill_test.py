#!/usr/bin/env python3
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv import cleanup, spill_stats
from data.visualize import sar_display
from inference.predict import load_model, prepare_image
from utils import PROJECT_ROOT, get_device, load_config

RAW = PROJECT_ROOT / "data" / "raw" / "web"
OUT = PROJECT_ROOT / "outputs" / "visualizations" / "web_test"
URL = (
    "https://www.esa.int/var/esa/storage/images/esa_multimedia/images/"
    "2018/10/mediterranean_slick/17770890-4-eng-GB/Mediterranean_slick.jpg"
)


def fetch(url, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "osiris"})
    dest.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return dest


def crop_ocean(path: Path, dest: Path) -> Path:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im.crop((int(w * 0.22), int(h * 0.08), int(w * 0.62), int(h * 0.55))).save(dest)
    return dest


def main():
    src = fetch(URL, RAW / "mediterranean_slick_esa.jpg")
    crop = crop_ocean(src, RAW / "mediterranean_slick_crop.png")
    cfg = load_config("configs/config.yaml")
    device = get_device()
    model = load_model(cfg, PROJECT_ROOT / "checkpoints" / "best.pt", device)
    OUT.mkdir(parents=True, exist_ok=True)

    sar, tensor = prepare_image(crop, cfg)
    amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
        logits = model(tensor.to(device))
    prob = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
    pred = (prob >= 0.5).astype(np.uint8)
    cv_mask = cleanup(prob)
    info = spill_stats(cv_mask, prob)

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4))
    for ax, name, arr in zip(
        axes,
        ["crop", "prob", "unet", "cv"],
        [np.clip(sar_display(sar), 0, 1), prob, pred, cv_mask],
    ):
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 and name != "prob" else ("inferno" if name == "prob" else None), vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle("Corsica 2018 Sentinel-1 (no gt)")
    fig.tight_layout()
    fig.savefig(OUT / "mediterranean_slick_overlay_panels.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(np.clip(sar_display(sar).astype(np.float32), 0, 1))
    ax.imshow(np.ma.masked_where(prob < 0.35, prob), cmap="YlOrRd", alpha=0.4, vmin=0.35, vmax=1)
    if pred.max():
        ax.contour(pred, levels=[0.5], colors=["#3ecbff"], linewidths=1, linestyles="--")
    if cv_mask.max():
        ax.contour(cv_mask, levels=[0.5], colors=["#2ee66b"], linewidths=1.6)
    ax.text(
        0.02,
        0.98,
        f"pred {int(info['area_pixels'])} px\nconf {float(info['confidence']):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="white",
        family="monospace",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 4, "edgecolor": "none"},
    )
    ax.legend(
        handles=[
            Patch(facecolor="#ffb347", alpha=0.45, label="prob"),
            Line2D([0], [0], color="#3ecbff", lw=1.5, ls="--", label="unet"),
            Line2D([0], [0], color="#2ee66b", lw=2, label="cv"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.7,
    )
    ax.set_title("Corsica 2018")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "mediterranean_slick_overlay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(OUT, "pred", info["area_pixels"])


if __name__ == "__main__":
    main()
