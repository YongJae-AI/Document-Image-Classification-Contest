#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.datamodule import create_dataloaders
from src.models.factory import create_model


ANGLE_BUCKETS: Dict[str, List[float]] = {
    "0": [0.0, -12.0, 12.0],
    "90": [90.0, 78.0, 102.0],
    "180": [180.0, 168.0, 192.0],
    "270": [270.0, 258.0, 282.0],
}

CONFUSION_PAIRS = [(3, 7), (7, 4), (13, 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation metrics per angle and class.")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path (defaults to <run_dir>/eval_metrics.json).",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: Dict, run_dir: Path) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg)
    ckpt_path = run_dir / "checkpoints" / "best.pth"
    state = torch.load(ckpt_path, map_location=device)
    model_state = (
        state.get("ema_state")
        or state.get("swa_state")
        or state.get("model_state")
        or state
    )
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def gather_predictions(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    preds = []
    probs = []
    targets = []
    ids = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            softmax = torch.softmax(outputs, dim=1)
            prob, pred = torch.max(softmax, dim=1)
            preds.append(pred.cpu().numpy())
            probs.append(prob.cpu().numpy())
            targets.append(labels.numpy())
            if isinstance(labels, torch.Tensor):
                ids.extend([None] * len(labels))
            else:
                ids.extend(labels)
    return {
        "preds": np.concatenate(preds),
        "probs": np.concatenate(probs),
        "targets": np.concatenate(targets),
    }


def rotate_batch(images: torch.Tensor, angle: float) -> torch.Tensor:
    if abs(((angle % 360) % 90)) < 1e-4:
        k = int(round(angle / 90.0)) % 4
        return torch.rot90(images, k=k, dims=(2, 3))
    return torch.stack(
        [
            F.rotate(
                img,
                float(angle),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            for img in images
        ],
        dim=0,
    )


def angle_bucket_metrics(model: torch.nn.Module, loader: torch.utils.data.DataLoader, targets: np.ndarray) -> Dict[str, float]:
    device = next(model.parameters()).device
    metrics = {}
    for bucket, angles in ANGLE_BUCKETS.items():
        bucket_preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                bucket_logits = []
                for angle in angles:
                    rotated = rotate_batch(images, angle)
                    outputs = model(rotated)
                    bucket_logits.append(torch.softmax(outputs, dim=1))
                mean_logits = torch.stack(bucket_logits, dim=0).mean(dim=0)
                pred = mean_logits.argmax(dim=1)
                bucket_preds.append(pred.cpu().numpy())
        bucket_preds = np.concatenate(bucket_preds)
        metrics[bucket] = f1_score(targets, bucket_preds, average="macro")
    return metrics


def class_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    class_f1 = f1_score(targets, preds, average=None)
    result = {str(idx): float(score) for idx, score in enumerate(class_f1)}
    result["macro"] = float(f1_score(targets, preds, average="macro"))
    return result


def confusion_pair_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, Dict[str, float]]:
    cm = confusion_matrix(targets, preds)
    pair_results = {}
    for a, b in CONFUSION_PAIRS:
        sub_targets_mask = np.isin(targets, [a, b])
        sub_preds = preds[sub_targets_mask]
        sub_targets = targets[sub_targets_mask]
        if len(sub_targets) == 0:
            pair_results[f"{a}-{b}"] = {"support": 0, "f1": None}
            continue
        f1 = f1_score(sub_targets, sub_preds, average="macro")
        pair_results[f"{a}-{b}"] = {"support": int(len(sub_targets)), "f1": float(f1)}
    return pair_results


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir)
    model = load_model(cfg, run_dir)

    (train_loader, valid_loader), _ = create_dataloaders(cfg)
    # reuse valid loader but ensure deterministic order
    base_results = gather_predictions(model, valid_loader)
    targets = base_results["targets"]

    angle_metrics = angle_bucket_metrics(model, valid_loader, targets)
    per_class = class_metrics(targets, base_results["preds"])
    pair_metrics = confusion_pair_metrics(targets, base_results["preds"])

    results = {
        "run_dir": str(run_dir),
        "macro_f1": per_class.pop("macro"),
        "class_f1": per_class,
        "angle_bucket_macro_f1": angle_metrics,
        "confusion_pairs": pair_metrics,
    }

    output_path = Path(args.output) if args.output else run_dir / "eval_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
