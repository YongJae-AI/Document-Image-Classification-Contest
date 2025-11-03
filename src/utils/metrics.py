from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score


class AverageMeter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


@dataclass
class MetricTracker:
    primary_metric: str = "f1"
    targets: List[int] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)

    def reset(self) -> None:
        self.targets.clear()
        self.predictions.clear()

    def update(self, targets: torch.Tensor, logits: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.targets.extend(targets.detach().cpu().numpy().tolist())
        self.predictions.extend(preds.detach().cpu().numpy().tolist())

    def compute(self) -> float:
        if not self.targets:
            return 0.0
        if self.primary_metric == "f1":
            return f1_score(
                np.array(self.targets),
                np.array(self.predictions),
                average="macro",
                zero_division=0,
            )
        raise ValueError(f"Unsupported metric: {self.primary_metric}")

