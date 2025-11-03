#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.datamodule import create_dataloaders
from src.models.factory import create_model
from src.training.engine import Trainer
from src.training.losses import create_loss
from src.training.optim import create_optimizer
from src.training.scheduler import create_scheduler
from src.utils.logging import create_logger
from src.utils.metrics import MetricTracker
from src.utils.performance import configure_performance
from src.utils.run_naming import build_run_name
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train document classifier.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Override fold index defined in config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.fold is not None:
        cfg["split"]["fold_index"] = args.fold
    if args.output_root is not None:
        cfg["paths"]["output_root"] = args.output_root

    seed_everything(
        cfg["experiment"]["seed"], cfg["experiment"].get("deterministic", True)
    )
    configure_performance(cfg)

    output_root = Path(cfg["paths"]["output_root"]).resolve()
    run_name = build_run_name(cfg)
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    logger = create_logger(run_dir / "train.log")
    logger.info("Starting run: %s", run_name)
    logger.info("Configuration:\n%s", json.dumps(cfg, indent=2))

    (train_loader, valid_loader), class_weights = create_dataloaders(cfg)
    model = create_model(cfg)
    if cfg.get("performance", {}).get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    criterion = create_loss(cfg, class_weights=class_weights)
    optimizer = create_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)
    metric_tracker = MetricTracker(primary_metric=cfg["training"]["save_best_metric"])

    trainer = Trainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_tracker=metric_tracker,
        logger=logger,
        run_dir=run_dir,
        channels_last=cfg.get("performance", {}).get("channels_last", False),
    )

    best_metric = trainer.fit(train_loader=train_loader, valid_loader=valid_loader)

    logger.info("Best %s: %.4f", cfg["training"]["save_best_metric"], best_metric)


if __name__ == "__main__":
    main()
