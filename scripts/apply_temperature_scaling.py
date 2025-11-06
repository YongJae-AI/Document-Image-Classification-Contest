#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import LBFGS
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.datamodule import create_dataloaders
from src.data.dataset import DocumentDataset
from src.models.factory import create_model
from src.transforms.factory import create_transforms
from src.utils.run_naming import build_submission_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply temperature scaling using validation logits and generate a calibrated submission."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing config/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint path. Defaults to run_dir/checkpoints/best.pth",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        default="data/raw/sample_submission.csv",
        help="Path to sample submission file.",
    )
    parser.add_argument(
        "--leaderboard-score",
        type=float,
        default=None,
        help="Leaderboard score to embed in filename.",
    )
    parser.add_argument(
        "--temperature-json",
        type=str,
        default=None,
        help="Optional path to save fitted temperature and calibration metrics.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="tscaled",
        help="Suffix appended to the generated submission filename.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> float:
    state = torch.load(checkpoint_path, map_location=device)
    metric = state.get("metric", 0.0)
    ema_state = state.get("ema_state")
    if ema_state:
        model.load_state_dict(ema_state, strict=False)
    else:
        swa_state = state.get("swa_state")
        if swa_state:
            model.load_state_dict(swa_state, strict=False)
        else:
            model_state = state.get("model_state") or state
            model.load_state_dict(model_state, strict=False)
    return metric


def gather_validation_logits(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            logits_list.append(outputs.detach())
            labels_list.append(labels.detach())
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return logits, labels


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        criterion = nn.CrossEntropyLoss()
        logits = logits.detach()
        labels = labels.detach()
        device = logits.device
        self.to(device)
        optimizer = LBFGS([self.log_temperature], lr=0.01, max_iter=50)

        def _closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)


def angle_list_from_cfg(cfg: Dict[str, Any]) -> List[float]:
    tta_cfg = cfg.get("submission", {}).get("tta", {})
    if not tta_cfg.get("enabled", False):
        return [0.0]
    rotations = tta_cfg.get("rotations", [0])
    jitter = float(tta_cfg.get("jitter_degrees", 0) or 0)
    angles: List[float] = []
    for base in rotations:
        base = float(base)
        angles.append(base)
        if jitter > 0:
            angles.extend([base - jitter, base + jitter])
    if not angles:
        angles = [0.0]
    normalized = [round(((angle + 180) % 360) - 180, 2) for angle in angles]
    unique = sorted(set(normalized))
    return unique


def apply_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    angles: List[float],
) -> torch.Tensor:
    device = images.device
    logits_stack: List[torch.Tensor] = []
    for angle in angles:
        if abs(angle) < 1e-4:
            rotated = images
        else:
            mod_angle = angle % 360
            if abs(mod_angle % 90) < 1e-4:
                k = int(round(mod_angle / 90)) % 4
                rotated = torch.rot90(images, k=k, dims=(2, 3))
            else:
                rotated = torch.stack(
                    [
                        F.rotate(
                            img,
                            angle,
                            interpolation=InterpolationMode.BILINEAR,
                            fill=0.0,
                        )
                        for img in images
                    ],
                    dim=0,
                )
        outputs = model(rotated)
        logits_stack.append(outputs)
    if len(logits_stack) == 1:
        return logits_stack[0]
    return torch.stack(logits_stack, dim=0).mean(dim=0)


def generate_calibrated_submission(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    temperature: float,
    sample_path: Path,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    test_transforms = create_transforms(cfg, is_train=False)
    sample_df = pd.read_csv(sample_path)
    test_dataset = DocumentDataset(
        dataframe=sample_df[["ID"]].copy(),
        image_dir=Path(cfg["paths"]["test_dir"]),
        transforms=test_transforms,
        is_train=False,
    )

    loader_cfg = cfg["data"]["loader"]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg.get("pin_memory", False),
        persistent_workers=loader_cfg.get("persistent_workers", False),
        prefetch_factor=(
            loader_cfg.get("prefetch_factor", 2)
            if loader_cfg.get("num_workers", 0) > 0
            else None
        ),
    )

    angles = angle_list_from_cfg(cfg)
    all_probs: List[np.ndarray] = []
    image_ids: List[str] = []
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, non_blocking=True)
            logits = apply_tta(model, images, angles)
            scaled_logits = logits / max(temperature, 1e-3)
            probs = torch.softmax(scaled_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            image_ids.extend(ids)

    return np.vstack(all_probs), image_ids


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else run_dir / "checkpoints" / "best.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg).to(device)
    metric_value = load_checkpoint(model, checkpoint_path, device)

    (_, valid_loader), _ = create_dataloaders(cfg)
    val_logits, val_labels = gather_validation_logits(model, valid_loader)

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)
    temperature = float(scaler.temperature.item())

    probs, image_ids = generate_calibrated_submission(
        model, cfg, temperature, Path(args.sample_submission), device
    )

    sample_df = pd.read_csv(args.sample_submission)
    assert list(sample_df["ID"]) == image_ids, "Prediction order mismatch with sample submission."
    preds = probs.argmax(axis=1)

    leaderboard_score = (
        args.leaderboard_score
        if args.leaderboard_score is not None
        else cfg["submission"].get("leaderboard_score", 0.0)
    )
    base_name = run_dir.name
    submission_name = f"{build_submission_name(cfg, metric_value, leaderboard_score)}_{args.suffix}"

    output_dir = Path("submissions")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{submission_name}.csv"
    submission_df = sample_df.copy()
    submission_df["target"] = preds
    submission_df.to_csv(csv_path, index=False)

    np.save(output_dir / f"{submission_name}.npy", probs)

    if args.temperature_json:
        report = {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "temperature": temperature,
            "validation_samples": int(val_labels.numel()),
            "metric_value": float(metric_value),
        }
        with Path(args.temperature_json).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Temperature: {temperature:.4f}")
    print(f"Saved calibrated submission to {csv_path}")


if __name__ == "__main__":
    main()
