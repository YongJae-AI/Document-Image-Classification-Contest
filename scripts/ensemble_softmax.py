#!/usr/bin/env python

import argparse
import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Softmax ensemble of submission logits.")
    parser.add_argument(
        "--logits",
        nargs="+",
        required=True,
        help="List of .npy probability files to ensemble.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights matching logits order (will be normalized).",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        default="data/raw/sample_submission.csv",
        help="Sample submission used to preserve ID ordering.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory to store ensembled submission.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional descriptor appended to the output filename.",
    )
    return parser.parse_args()


def load_logits(paths: List[str]) -> List[np.ndarray]:
    arrays = [np.load(p) for p in paths]
    base_shape = arrays[0].shape
    for path, arr in zip(paths, arrays):
        if arr.shape != base_shape:
            raise ValueError(f"Logit shape mismatch for {path}: expected {base_shape}, got {arr.shape}")
    return arrays


def main() -> None:
    args = parse_args()
    logits_arrays = load_logits(args.logits)

    if args.weights:
        if len(args.weights) != len(logits_arrays):
            raise ValueError("Number of weights must match number of logits files.")
        weights = np.array(args.weights, dtype=np.float64)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(logits_arrays), dtype=np.float64) / len(logits_arrays)

    stacked = np.stack(logits_arrays, axis=0)
    weighted_probs = np.tensordot(weights, stacked, axes=(0, 0))

    preds = weighted_probs.argmax(axis=1)

    sample_df = pd.read_csv(args.sample_submission)
    if len(sample_df) != len(preds):
        raise ValueError("Sample submission and logits length mismatch.")

    submission_df = sample_df.copy()
    submission_df["target"] = preds

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    strategy = args.tag or ("weighted" if args.weights else "uniform")
    output_dir = Path(args.output_root) / "ensembles"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{timestamp}-ensemble_{strategy}.csv"
    submission_df.to_csv(csv_path, index=False)

    probs_path = output_dir / f"{timestamp}-ensemble_{strategy}.npy"
    np.save(probs_path, weighted_probs)

    print(f"Saved ensemble CSV to {csv_path}")


if __name__ == "__main__":
    main()
