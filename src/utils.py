from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
import torch

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
PROJECT_ROOT = ROOT
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    if not p.is_absolute():
        p = ROOT / p
    with p.open() as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    rel = Path(cfg["paths"][key])
    return rel if rel.is_absolute() else ROOT / rel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_cuda() -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.90)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
