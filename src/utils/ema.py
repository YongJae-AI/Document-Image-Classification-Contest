from __future__ import annotations

from typing import Dict

import torch


class ModelEma:
    """Exponential Moving Average for model parameters/buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = decay
        self._device = next(model.parameters()).device
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        self.shadow = {k: v.detach().clone() for k, v in state.items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        decay = self.decay
        state = model.state_dict()
        for key, param in state.items():
            if key not in self.shadow:
                self.shadow[key] = param.detach().clone()
                continue
            shadow_param = self.shadow[key]
            if torch.is_floating_point(param):
                shadow_param.mul_(decay).add_(param.detach() * (1.0 - decay))
            else:
                shadow_param.copy_(param)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        self.backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        model.load_state_dict(self.backup, strict=False)
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}
