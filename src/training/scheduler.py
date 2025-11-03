from typing import Any, Dict

import torch


def create_scheduler(
    cfg: Dict[str, Any], optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler_cfg = cfg["scheduler"]
    name = scheduler_cfg.get("name", "cosine").lower()
    epochs = cfg["training"]["epochs"]

    if name == "cosine":
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)
        min_lr = scheduler_cfg.get("min_lr", 0.0)
        cosine_epochs = max(1, epochs - warmup_epochs)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=min_lr,
        )

        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

        return cosine_scheduler

    raise ValueError(f"Unsupported scheduler: {name}")

