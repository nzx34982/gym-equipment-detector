from __future__ import annotations

import random

import numpy as np
import torch


def collate_fn(batch):
    """Keep variable-length detection targets as a tuple of samples."""
    return tuple(zip(*batch))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable small experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
