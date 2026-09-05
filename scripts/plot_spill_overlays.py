#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv import cleanup, spill_stats
from data.io import load_mask
from data.preprocess import binarize_mask, resize_pair
from data.visualize import sar_display
from inference.predict import load_model, prepare_image
from training.metrics import metrics
from utils import PROJECT_ROOT, get_device, load_config

OUT = PROJECT_ROOT / "outputs" / "visualizations" / "spill_test"
N = 12


def pick_pairs(split_file: Path, n: int):
    rows = []
    for line in split_file.read_text().splitlines():
        img_s, msk_s = line.split("\t")
        img, msk = Path(img_s), Path(msk_s)
        pos = int((load_mask(msk) > 0).sum())
        rows.append((pos, img, msk))
    rows.sort(reverse=True)
    oil = [r for r in rows if 200 < r[0] < 40000]
    empty = [r for r in rows if r[0] == 0]
    sent = [r for r in oil if "sentinel" in r[1].name]
    pal = [r for r in oil if "palsar" in r[1].name]
    picked = sent[: n // 2] + pal[: max(n // 2 - 1, 1)] + empty[:1]
    return [(img, msk, pos) for pos, img, msk in picked[:n]]


def draw(ax, sar, gt, pred, cv_mask, prob, title, stats):
    ax.imshow(np.clip(sar_display(sar).astype(np.float32), 0, 1))
    ax.imshow(np.ma.masked_where(prob < 0.35, prob), cmap="YlOrRd", alpha=0.35, vmin=0.35, vmax=1)
    if cv_mask.max():
        ax.contour(cv_mask, levels=[0.5], colors=["#2ee66b"], linewidths=1.6)
    if pred.max():
        ax.contour(pred, levels=[0.5], colors=["#3ecbff"], linewidths=1, linestyles="--")
    if gt.max():
        ax.contour(gt, levels=[0.5], colors=["#ff3355"], linewidths=1.4)
    ax.text(
        0.02,
        0.98,
        f"dice {stats['dice']:.3f}\npred {stats['pred_px']} px\ngt {stats['gt_px']} px\nconf {stats['conf']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="white",
        family="monospace",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 4, "edgecolor": "none"},
    )
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def main():
    cfg = load_config("configs/config.yaml")
    device = get_device()
    model = load_model(cfg, PROJECT_ROOT / "checkpoints" / "best.pt", device)
    pairs = pick_pairs(PROJECT_ROOT / "data" / "splits" / "test.txt", N)
    OUT.mkdir(parents=True, exist_ok=True)
    size = int(cfg["dataset"]["image_size"])
    amp = device.type == "cuda"
    rows = []

    for img_p, msk_p, _ in pairs:
        sar, tensor = prepare_image(img_p, cfg)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
            logits = model(tensor.to(device))
        prob = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
        gt = binarize_mask(load_mask(msk_p))
        _, gt = resize_pair(sar, gt, size)
        pred = (prob >= 0.5).astype(np.uint8)
        cv_mask = cleanup(prob)
        dice = metrics(logits.float().cpu(), torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float())["dice"]
        info = spill_stats(cv_mask, prob)
        stats = {"dice": dice, "pred_px": int(info["area_pixels"]), "gt_px": int(gt.sum()), "conf": float(info["confidence"])}

        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        draw(ax, sar, gt, pred, cv_mask, prob, img_p.name, stats)
        ax.legend(
            handles=[
                Patch(facecolor="#ffb347", alpha=0.45, label="prob"),
                Line2D([0], [0], color="#ff3355", lw=2, label="gt"),
                Line2D([0], [0], color="#3ecbff", lw=1.5, ls="--", label="unet"),
                Line2D([0], [0], color="#2ee66b", lw=2, label="cv"),
            ],
            loc="lower right",
            fontsize=7,
            framealpha=0.7,
        )
        dest = OUT / f"{img_p.stem}_overlay.png"
        fig.tight_layout()
        fig.savefig(dest, dpi=160, bbox_inches="tight")
        plt.close(fig)
        rows.append((sar, gt, pred, cv_mask, prob, img_p.name, stats))
        print(dest.name, f"dice={dice:.3f}")

    cols = 4
    r = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(r, cols, figsize=(4.2 * cols, 4.4 * r))
    axes = np.atleast_2d(axes)
    for i, item in enumerate(rows):
        rr, cc = divmod(i, cols)
        draw(axes[rr, cc], *item)
    for j in range(len(rows), r * cols):
        rr, cc = divmod(j, cols)
        axes[rr, cc].axis("off")
    fig.suptitle("held-out SOS  (red=gt, cyan=unet, green=cv)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / "gallery.png")


if __name__ == "__main__":
    main()
