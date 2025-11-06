from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names


def create_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optimizer"]
    name = optim_cfg.get("name", "adamw").lower()

    parameters = _build_parameter_groups(model, optim_cfg)

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


def _build_parameter_groups(model: torch.nn.Module, optim_cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    lr_scale = optim_cfg.get("lr_scale")
    layer_decay = optim_cfg.get("layer_decay")
    if not layer_decay and not lr_scale:
        return [p for p in model.parameters() if p.requires_grad]

    base_lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    decay_params = get_parameter_names(model, [nn.LayerNorm])
    decay_params = [name for name in decay_params if "bias" not in name]

    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_hidden_layers = int(model.config.num_hidden_layers)
    else:
        num_hidden_layers = 12

    total_layers = num_hidden_layers + 2

    manual_scales: List[float] = []
    if isinstance(lr_scale, (list, tuple)):
        manual_scales = [float(x) for x in lr_scale]

    layer_decay = float(layer_decay) if layer_decay is not None else None

    def resolve_scale(layer_idx: int) -> float:
        if manual_scales:
            return manual_scales[min(layer_idx, len(manual_scales) - 1)]
        if layer_decay is not None:
            return layer_decay ** (total_layers - 1 - layer_idx)
        return 1.0

    def layer_id_from_name(name: str) -> int:
        if name.startswith("beit.embeddings"):
            return 0
        if "encoder.layer." in name:
            try:
                layer = int(name.split("encoder.layer.")[1].split(".")[0])
                return layer + 1
            except ValueError:
                return total_layers - 2
        if name.startswith("beit.encoder.layernorm"):
            return total_layers - 2
        return total_layers - 1

    param_groups: Dict[Tuple[int, bool], Dict[str, Any]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_decay = name in decay_params
        layer_id = layer_id_from_name(name)
        scale = resolve_scale(layer_id)
        key = (layer_id, use_decay)
        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": weight_decay if use_decay else 0.0,
            }
        param_groups[key]["params"].append(param)

    return list(param_groups.values())
