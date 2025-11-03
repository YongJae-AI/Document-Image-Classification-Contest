from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def create_loss(
    cfg: Dict[str, Any], class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    loss_cfg = cfg["loss"]
    name = loss_cfg.get("name", "cross_entropy")

    if name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {name}")

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)

    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )

