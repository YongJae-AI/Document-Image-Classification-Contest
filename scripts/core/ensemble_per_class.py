#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description='Per-class weighted ensemble of submissions')
    ap.add_argument('--sample-sub', default='data/raw/sample_submission.csv')
    ap.add_argument('--probs', nargs='+', required=True, help='List of .npy probability files (same order as --models)')
    ap.add_argument('--models', nargs='+', required=True, help='Model aliases matching probs order (e.g., effnet_b7 swin_large convnextv2 effnetv2_l)')
    ap.add_argument('--weights-json', type=str, required=False, help='JSON mapping {"class_index": {"model": weight, ...}, ...}')
    ap.add_argument('--out-csv', type=str, required=True)
    return ap.parse_args()


def load_weights(models: List[str], n_classes: int, weights_json: Path | None) -> np.ndarray:
    # weights shape: (n_models, n_classes)
    W = np.ones((len(models), n_classes), dtype=np.float32) / float(len(models))
    if weights_json is None:
        return W
    data: Dict[str, Dict[str, float]] = json.loads(Path(weights_json).read_text())
    for cls_str, m2w in data.items():
        c = int(cls_str)
        s = 0.0
        for mi, m in enumerate(models):
            w = float(m2w.get(m, 0.0))
            W[mi, c] = w
            s += w
        if s > 0:
            W[:, c] /= s  # normalize per-class
    return W


def main():
    args = parse_args()
    probs_paths = [Path(p) for p in args.probs]
    arrs = [np.load(p) for p in probs_paths]
    n = arrs[0].shape[0]
    n_classes = arrs[0].shape[1]
    for a in arrs[1:]:
        assert a.shape == (n, n_classes), 'All probs must have same shape'

    W = load_weights(args.models, n_classes, Path(args.weights_json) if args.weights_json else None)
    # Ensemble: for each class c, combine model probs with weights[:, c]
    ens = np.zeros_like(arrs[0], dtype=np.float32)
    for mi, a in enumerate(arrs):
        for c in range(n_classes):
            ens[:, c] += W[mi, c] * a[:, c]
    preds = ens.argmax(axis=1)

    df_sample = pd.read_csv(args.sample_sub)
    out = df_sample.copy()
    out['target'] = preds
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print('Saved ensemble submission to', args.out_csv)


if __name__ == '__main__':
    main()

