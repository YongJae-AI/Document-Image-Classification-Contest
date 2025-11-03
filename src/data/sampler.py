from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def build_sampler(labels: np.ndarray, sampler_cfg: dict) -> Optional[Sampler]:
    if sampler_cfg is None:
        return None

    name = sampler_cfg.get("name", "")
    if name != "weighted_random":
        return None

    replacement = sampler_cfg.get("replacement", True)
    minority_classes = set(sampler_cfg.get("minority_classes", []))
    power = float(sampler_cfg.get("weights_power", 1.0))

    label_series = pd.Series(labels)
    class_counts = label_series.value_counts().to_dict()
    max_count = max(class_counts.values())

    sample_weights = []
    for label in labels:
        if minority_classes and label not in minority_classes:
            sample_weights.append(1.0)
            continue
        count = class_counts.get(label, 1)
        weight = (max_count / count) ** power
        sample_weights.append(weight)

    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(labels), replacement=replacement)

