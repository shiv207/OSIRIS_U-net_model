from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * targets).sum(dim=dims)
    den = p.sum(dim=dims) + targets.sum(dim=dims)
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


class SegLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        loss = logits.new_zeros(())
        if self.bce_w:
            loss = loss + self.bce_w * F.binary_cross_entropy_with_logits(logits, targets)
        if self.dice_w:
            loss = loss + self.dice_w * dice_loss(logits, targets)
        return loss
