from __future__ import annotations

import torch


@torch.no_grad()
def metrics(logits, targets, thresh=0.5, eps=1e-7):
    pred = (torch.sigmoid(logits) >= thresh).float()
    tgt = (targets >= 0.5).float()
    tp = (pred * tgt).sum()
    fp = (pred * (1 - tgt)).sum()
    fn = ((1 - pred) * tgt).sum()
    tn = ((1 - pred) * (1 - tgt)).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return {
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec + eps)),
        "pixel_acc": float((tp + tn) / (tp + tn + fp + fn + eps)),
    }


class AverageMeter:
    def __init__(self):
        self.sums = {}
        self.n = 0

    def update(self, vals: dict, loss=None):
        for k, v in vals.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
        if loss is not None:
            self.sums["loss"] = self.sums.get("loss", 0.0) + loss
        self.n += 1

    def mean(self):
        if not self.n:
            return {}
        return {k: v / self.n for k, v in self.sums.items()}
