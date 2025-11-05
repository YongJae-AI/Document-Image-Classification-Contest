import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .dataset import DocumentDataset
from .sampler import build_sampler
from src.transforms.factory import create_transforms


def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[Tuple[DataLoader, DataLoader], None]:
    image_dir = Path(cfg["paths"]["image_dir"])

    dataframe = pd.read_csv(cfg["paths"]["train_csv"])

    excluded_ids = set(cfg["data"].get("excluded_ids", []) or [])
    if excluded_ids:
        dataframe = dataframe[~dataframe["ID"].isin(excluded_ids)].reset_index(drop=True)

    predefined_split = cfg["split"].get("predefined_split")
    train_indices, valid_indices = None, None
    if predefined_split:
        split_path = Path(predefined_split)
        if not split_path.exists():
            raise FileNotFoundError(f"Predefined split file not found: {split_path}")
        with split_path.open("rb") as f:
            splits = pickle.load(f)
        if not (0 <= cfg["split"]["fold_index"] < len(splits)):
            raise ValueError(
                f"Fold index {cfg['split']['fold_index']} out of range for predefined splits."
            )
        train_indices, valid_indices = splits[cfg["split"]["fold_index"]]
    else:
        splitter = StratifiedKFold(
            n_splits=cfg["split"]["n_splits"],
            shuffle=True,
            random_state=cfg["split"]["seed"],
        )
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(dataframe["ID"], dataframe["target"])
        ):
            if fold == cfg["split"]["fold_index"]:
                train_indices, valid_indices = train_idx, valid_idx
                break

    if train_indices is None or valid_indices is None:
        raise ValueError(f"Invalid fold index: {cfg['split']['fold_index']}")

    train_df = dataframe.iloc[train_indices].reset_index(drop=True)
    valid_df = dataframe.iloc[valid_indices].reset_index(drop=True)

    train_transforms = create_transforms(cfg, is_train=True)
    valid_transforms = create_transforms(cfg, is_train=False)

    train_dataset = DocumentDataset(
        dataframe=train_df,
        image_dir=image_dir,
        transforms=train_transforms,
        is_train=True,
    )
    valid_dataset = DocumentDataset(
        dataframe=valid_df,
        image_dir=image_dir,
        transforms=valid_transforms,
        is_train=True,
    )

    sampler = build_sampler(train_df["target"].values, cfg.get("sampler"))
    loader_cfg = cfg["data"]["loader"]

    prefetch_factor = loader_cfg.get("prefetch_factor", 2)
    if loader_cfg.get("num_workers", 0) == 0:
        prefetch_factor = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg.get("pin_memory", False),
        persistent_workers=loader_cfg.get("persistent_workers", False),
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg.get("pin_memory", False),
        persistent_workers=loader_cfg.get("persistent_workers", False),
        prefetch_factor=prefetch_factor,
    )

    return (train_loader, valid_loader), None
