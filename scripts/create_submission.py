#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.dataset import DocumentDataset
from src.models.factory import create_model
from src.transforms.factory import create_transforms
from src.utils.run_naming import build_submission_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission CSV.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config (optional if --run-dir provided).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Training run directory containing config/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
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
        "--save-probs",
        action="store_true",
        help="Save softmax probabilities alongside CSV.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.run_dir:
        config_path = Path(args.run_dir) / "config.yaml"
    elif args.config:
        config_path = Path(args.config)
    else:
        raise ValueError("Either --run-dir or --config must be provided.")

    with open(config_path, "r", encoding="utf-8") as f:
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    run_dir = Path(args.run_dir) if args.run_dir else None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is None:
        if run_dir is None:
            raise ValueError("Provide --checkpoint when --run-dir is absent.")
        checkpoint_path = run_dir / "checkpoints" / "best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(cfg)
    metric_value = load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    test_transforms = create_transforms(cfg, is_train=False)

    sample_df = pd.read_csv(args.sample_submission)
    test_dataset = DocumentDataset(
        dataframe=sample_df[["ID"]].copy(),
        image_dir=Path(cfg["paths"]["test_dir"]),
        transforms=test_transforms,
        is_train=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["data"]["loader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["loader"]["num_workers"],
        pin_memory=cfg["data"]["loader"].get("pin_memory", False),
        persistent_workers=cfg["data"]["loader"].get("persistent_workers", False),
        prefetch_factor=(
            cfg["data"]["loader"].get("prefetch_factor", 2)
            if cfg["data"]["loader"].get("num_workers", 0) > 0
            else None
        ),
    )

    all_probs = []
    image_ids = []
    tta_cfg = cfg.get("submission", {}).get("tta", {})
    tta_enabled = tta_cfg.get("enabled", False)
    rotations = tta_cfg.get("rotations", [0]) if tta_enabled else [0]
    jitter = float(tta_cfg.get("jitter_degrees", 0) or 0)

    tta_angles = []
    for base in rotations:
        base = float(base)
        tta_angles.append(base)
        if jitter > 0:
            tta_angles.extend([base - jitter, base + jitter])
    if not tta_angles:
        tta_angles = [0.0]
    # Normalize angles to range [-180, 180) to avoid redundant rotations
    normalized_angles = []
    for angle in tta_angles:
        a = ((angle + 180) % 360) - 180
        normalized_angles.append(round(a, 2))
    unique_angles = sorted(set(normalized_angles))

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, non_blocking=True)
            tta_probs = []
            for angle in unique_angles:
                if abs(angle) < 1e-4:
                    rotated = images
                else:
                    mod_angle = angle % 360
                    # Use fast integer rotations when possible
                    if abs((mod_angle % 90)) < 1e-4:
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
                tta_probs.append(torch.softmax(outputs, dim=1))
            if len(tta_probs) == 1:
                probs = tta_probs[0]
            else:
                probs = torch.stack(tta_probs, dim=0).mean(dim=0)
            all_probs.append(probs.cpu().numpy())
            image_ids.extend(ids)

    probs_array = np.vstack(all_probs)
    preds = probs_array.argmax(axis=1)

    assert list(sample_df["ID"]) == image_ids, "Prediction order does not match sample submission."

    submission_df = sample_df.copy()
    submission_df["target"] = preds

    leaderboard_score = (
        args.leaderboard_score
        if args.leaderboard_score is not None
        else cfg["submission"].get("leaderboard_score", 0.0)
    )
    base_name = run_dir.name if run_dir is not None else None
    submission_name = build_submission_name(
        cfg,
        metric_value,
        leaderboard_score,
        base_name=base_name,
    )

    output_dir = Path(cfg["paths"]["output_root"]) / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{submission_name}.csv"
    submission_df.to_csv(csv_path, index=False)

    if args.save_probs:
        probs_dir = Path(cfg["paths"]["output_root"]) / "logits"
        probs_dir.mkdir(parents=True, exist_ok=True)
        np.save(probs_dir / f"{submission_name}.npy", probs_array)

    print(f"Saved submission to {csv_path}")


if __name__ == "__main__":
    main()
