from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import make_loader
from models.losses import SegLoss
from models.unet import UNet
from training.metrics import AverageMeter, metrics
from training.validate import validate_epoch
from utils import ROOT, get_device, load_config, resolve_path, set_seed, setup_cuda, setup_logging

log = logging.getLogger(__name__)


def train_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    meter = AverageMeter()
    use_amp = amp and device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        meter.update(metrics(logits.detach().float(), y), loss=float(loss.detach()))
    return meter.mean()


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    setup_cuda()
    device = get_device()
    log.info("device %s", device)

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val", shuffle=False)
    model = UNet(
        int(cfg["dataset"]["in_channels"]),
        int(cfg["model"]["base_channels"]),
        int(cfg["model"]["depth"]),
    ).to(device)
    criterion = SegLoss(**cfg["loss"]).to(device)
    opt = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sch = cfg["train"]["scheduler"]
    scheduler = ReduceLROnPlateau(
        opt, mode="max", factor=float(sch["factor"]), patience=int(sch["patience"]), min_lr=float(sch["min_lr"])
    )
    amp = bool(cfg["train"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    ckpt_dir = resolve_path(cfg, "checkpoint_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolve_path(cfg, "log_dir") / "train_log.csv"
    start, best, wait = 0, -1.0, int(cfg["train"]["early_stopping_patience"])

    resume = args.resume or cfg["train"].get("resume")
    if resume:
        resume = Path(resume)
        if not resume.is_absolute():
            resume = ROOT / resume
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start = int(ckpt["epoch"]) + 1
        best = float(ckpt.get("best_dice", -1))
        log.info("resume %s (epoch %d)", resume, start)

    for epoch in range(start, int(cfg["train"]["epochs"])):
        tr = train_epoch(model, train_loader, criterion, opt, scaler, device, amp)
        va = validate_epoch(model, val_loader, criterion, device, amp)
        scheduler.step(va["dice"])
        lr = opt.param_groups[0]["lr"]
        log.info(
            "epoch %d  lr=%.1e  train dice=%.4f  val dice=%.4f iou=%.4f",
            epoch,
            lr,
            tr["dice"],
            va["dice"],
            va["iou"],
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        new_file = not log_path.exists()
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "best_dice": best,
            "config": cfg,
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if va["dice"] > best:
            best = va["dice"]
            payload["best_dice"] = best
            torch.save(payload, ckpt_dir / "best.pt")
            log.info("best dice %.4f", best)
            wait = int(cfg["train"]["early_stopping_patience"])
        else:
            wait -= 1
            if wait <= 0:
                log.info("early stop at %d (best %.4f)", epoch, best)
                break

    log.info("done  best=%.4f", best)


if __name__ == "__main__":
    main()
