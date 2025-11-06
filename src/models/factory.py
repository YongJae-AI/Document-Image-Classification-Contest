from typing import Any, Dict

import timm
import torch
from transformers import AutoConfig, AutoModelForImageClassification


def create_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    num_classes = cfg["data"]["num_classes"]

    if model_cfg.get("provider", "timm") == "huggingface":
        hf_name = model_cfg["hf_name"]
        config = AutoConfig.from_pretrained(hf_name, num_labels=num_classes, finetuning_task="image-classification")
        img_size = model_cfg.get("image_size")
        if img_size:
            config.image_size = img_size
        model = AutoModelForImageClassification.from_pretrained(
            hf_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        if model_cfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
    else:
        model_kwargs = {
            "pretrained": model_cfg.get("pretrained", True),
            "num_classes": num_classes,
            "drop_path_rate": model_cfg.get("drop_path_rate", 0.0),
        }
        if model_cfg.get("use_checkpoint"):
            model_kwargs["use_checkpoint"] = True

        model = timm.create_model(
            model_cfg["name"],
            **model_kwargs,
        )

    checkpoint_path = model_cfg.get("checkpoint")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        key = "state_dict" if "state_dict" in state else None
        model.load_state_dict(state[key] if key else state, strict=False)

    return model
