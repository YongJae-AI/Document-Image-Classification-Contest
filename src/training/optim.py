from typing import Any, Dict

import torch


def create_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optimizer"]
    name = optim_cfg.get("name", "adamw").lower()

    parameters = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=optim_cfg["lr"],
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=optim_cfg["lr"],
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=optim_cfg["lr"],
            momentum=optim_cfg.get("momentum", 0.9),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
            nesterov=optim_cfg.get("nesterov", False),
        )

    raise ValueError(f"Unsupported optimizer: {name}")
