from typing import Any, Dict

import timm
import torch


def create_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    num_classes = cfg["data"]["num_classes"]

    model = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg.get("pretrained", True),
        num_classes=num_classes,
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )

    checkpoint_path = model_cfg.get("checkpoint")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        key = "state_dict" if "state_dict" in state else None
        model.load_state_dict(state[key] if key else state, strict=False)

    return model

