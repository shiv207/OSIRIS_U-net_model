from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import make_loader
from models.losses import SegLoss
from models.unet import UNet
from training.metrics import AverageMeter, metrics
from utils import get_device, load_config, setup_logging

log = logging.getLogger(__name__)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, amp, thresh=0.5):
    model.eval()
    meter = AverageMeter()
    use_amp = amp and device.type == "cuda"
    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        meter.update(metrics(logits.float(), y, thresh), loss=float(loss.detach()))
    return meter.mean()


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = p.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    model = UNet(
        int(cfg["dataset"]["in_channels"]),
        int(cfg["model"]["base_channels"]),
        int(cfg["model"]["depth"]),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    stats = validate_epoch(
        model,
        make_loader(cfg, args.split, shuffle=False),
        SegLoss(**cfg["loss"]),
        device,
        bool(cfg["train"]["amp"]),
    )
    log.info(" ".join(f"{k}={v:.4f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
