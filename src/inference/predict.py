from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv import cleanup
from data.io import load_image
from data.preprocess import match_channels, normalize_sar, resize_pair
from data.visualize import save_panel
from models.unet import UNet
from utils import get_device, load_config, resolve_path, setup_logging

log = logging.getLogger(__name__)


def load_model(cfg, checkpoint: Path, device):
    model = UNet(
        int(cfg["dataset"]["in_channels"]),
        int(cfg["model"]["base_channels"]),
        int(cfg["model"]["depth"]),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def prepare_image(path: Path, cfg):
    image = normalize_sar(load_image(path))
    dummy = np.zeros(image.shape[:2], np.uint8)
    image, _ = resize_pair(image, dummy, int(cfg["dataset"]["image_size"]))
    image = match_channels(image, int(cfg["dataset"]["in_channels"]))
    t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0)
    return image, t


@torch.no_grad()
def predict_image(cfg, image_path: Path, checkpoint: Path | None = None):
    device = get_device()
    ckpt_dir = resolve_path(cfg, "checkpoint_dir")
    checkpoint = checkpoint or ckpt_dir / "best.pt"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = load_model(cfg, checkpoint, device)
    sar, tensor = prepare_image(image_path, cfg)
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=amp):
        logits = model(tensor.to(device))
    prob = torch.sigmoid(logits.float()).squeeze().cpu().numpy()

    inf = cfg.get("inference") or {}
    thresh = float(inf.get("threshold", 0.5))
    pred = (prob >= thresh).astype(np.uint8)
    cv_cfg = cfg.get("cv") or {}
    cv_mask = pred
    if inf.get("apply_cv_postprocess", True):
        cv_mask = cleanup(
            prob,
            thresh,
            int(cv_cfg.get("morph_open_ksize", 3)),
            int(cv_cfg.get("morph_close_ksize", 5)),
            int(inf.get("min_region_pixels", cv_cfg.get("min_component_pixels", 32))),
        )

    out = resolve_path(cfg, "inference_dir")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{image_path.stem}_panel.png"
    save_panel(dest, sar, pred=pred.astype(np.float32), cv_pred=cv_mask.astype(np.float32), title=image_path.name)
    log.info("%s -> %s", image_path.name, dest)
    return dest


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    predict_image(load_config(args.config), Path(args.image), Path(args.checkpoint) if args.checkpoint else None)


if __name__ == "__main__":
    main()
