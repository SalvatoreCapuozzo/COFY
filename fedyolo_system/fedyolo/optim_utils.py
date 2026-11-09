"""Tiny shared training utilities."""

from __future__ import annotations

import torch


def schedule_factor(cfg, progress: float) -> float:
    """Learning-rate multiplier at `progress` (0..1) of the whole training budget:
    1 for lr_schedule "constant"; 1 -> lrf linearly for "linear"."""
    if cfg.federation.lr_schedule == "linear":
        return 1.0 - (1.0 - cfg.federation.lrf) * min(max(progress, 0.0), 1.0)
    if cfg.federation.lr_schedule != "constant":
        raise ValueError(f"unknown lr_schedule {cfg.federation.lr_schedule!r}")
    return 1.0


def set_warmup_lr(optimizer: torch.optim.Optimizer, step: int, warmup_steps: int, lr0: float,
                  factor: float = 1.0) -> None:
    """Linear LR warmup from 0 -> lr0 over the first `warmup_steps` steps.

    Random-init detectors take large, unstable gradient steps on the very
    first few batches (box/DFL predictions start essentially arbitrary);
    skipping warmup is the most common cause of the loss oscillating wildly
    instead of decreasing, especially with tiny batch sizes.
    """
    lr = lr0 * factor * min(1.0, (step + 1) / max(1, warmup_steps))
    for g in optimizer.param_groups:
        g["lr"] = lr
