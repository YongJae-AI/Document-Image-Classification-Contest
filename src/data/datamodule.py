import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.multiprocessing as mp
import cv2
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from .dataset import DocumentDataset
from .sampler import build_sampler
from src.transforms.factory import create_transforms


def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[Tuple[DataLoader, DataLoader], None]:
    # Reduce thread contention that can cause DataLoader stalls
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    image_dir = Path(cfg["paths"]["image_dir"])

    dataframe = pd.read_csv(cfg["paths"]["train_csv"])

    excluded_ids = set(cfg["data"].get("excluded_ids", []) or [])
    if excluded_ids:
        dataframe = dataframe[~dataframe["ID"].isin(excluded_ids)].reset_index(drop=True)

    # Support full-train mode (train on all, with optional small holdout for monitoring)
    split_cfg = cfg.get("split", {})
    if bool(split_cfg.get("full_train", False)):
        holdout_frac = float(split_cfg.get("full_train_holdout_fraction", 0.02) or 0.0)
        seed = int(split_cfg.get("seed", 2024))
        if holdout_frac > 0.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
            (train_indices, valid_indices), = sss.split(dataframe["ID"], dataframe["target"])
        else:
            # No holdout: use entire dataset for both train/valid (validation will reflect training performance)
            train_indices = dataframe.index.values
            valid_indices = dataframe.index.values
    else:
        predefined_split = split_cfg.get("predefined_split")
        train_indices, valid_indices = None, None
        if predefined_split:
            split_path = Path(predefined_split)
            if not split_path.exists():
                raise FileNotFoundError(f"Predefined split file not found: {split_path}")
            with split_path.open("rb") as f:
                splits = pickle.load(f)
            if not (0 <= split_cfg["fold_index"] < len(splits)):
                raise ValueError(
                    f"Fold index {split_cfg['fold_index']} out of range for predefined splits."
                )
            block = splits[split_cfg["fold_index"]]
            if isinstance(block, (list, tuple)) and len(block) >= 2:
                train_indices, valid_indices = block[0], block[1]
            elif isinstance(block, dict):
                train_indices = block.get("train") or block.get("train_idx")
                valid_indices = block.get("val") or block.get("valid") or block.get("val_idx")
        else:
            splitter = StratifiedKFold(
                n_splits=split_cfg["n_splits"],
                shuffle=True,
                random_state=split_cfg["seed"],
            )
            for fold, (train_idx, valid_idx) in enumerate(
                splitter.split(dataframe["ID"], dataframe["target"])
            ):
                if fold == split_cfg["fold_index"]:
                    train_indices, valid_indices = train_idx, valid_idx
                    break

        if train_indices is None or valid_indices is None:
            raise ValueError(f"Invalid fold index: {split_cfg.get('fold_index')}")

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
    timeout = int(loader_cfg.get("timeout", 0))
    mp_ctx_name = loader_cfg.get("mp_context", None)
    mp_ctx = mp.get_context(mp_ctx_name) if mp_ctx_name else None

    prefetch_factor = loader_cfg.get("prefetch_factor", 2)
    if loader_cfg.get("num_workers", 0) == 0:
        prefetch_factor = None

    common_kwargs = {}
    if timeout:
        common_kwargs["timeout"] = timeout
    if mp_ctx is not None:
        common_kwargs["multiprocessing_context"] = mp_ctx

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
        **common_kwargs,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg.get("pin_memory", False),
        persistent_workers=loader_cfg.get("persistent_workers", False),
        prefetch_factor=prefetch_factor,
        **common_kwargs,
    )

    return (train_loader, valid_loader), None
