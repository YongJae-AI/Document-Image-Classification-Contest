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
    if name == "steplr":
        step_size = scheduler_cfg.get("step_size", max(1, epochs // 3))
        gamma = scheduler_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    if name == "multistep":
        milestones = scheduler_cfg.get("milestones")
        gamma = scheduler_cfg.get("gamma", 0.1)
        if not milestones:
            # Default: drop twice at 50% and 75% of total epochs
            milestones = [int(epochs * 0.5), int(epochs * 0.75)]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    raise ValueError(f"Unsupported scheduler: {name}")
