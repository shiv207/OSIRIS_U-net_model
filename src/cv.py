from __future__ import annotations

import cv2
import numpy as np


def cleanup(prob, thresh=0.5, open_k=3, close_k=5, min_px=32):
    if prob.ndim == 3:
        prob = np.squeeze(prob)
    mask = (prob >= thresh).astype(np.uint8)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[labels == i] = 1
    return keep


def spill_stats(mask, prob=None):
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    area = int(mask.astype(bool).sum())
    conf = 0.0
    if prob is not None and area:
        conf = float(np.squeeze(prob)[mask.astype(bool)].mean())
    elif prob is not None:
        conf = float(np.squeeze(prob).max())
    return {"area_pixels": area, "confidence": conf}
