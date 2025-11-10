#!/usr/bin/env python
"""
Auto TTA submit with cache + optional temperature scaling + report update.

Usage:
  python scripts/auto_tta_submit.py \
    --run-dir outputs/runs/<run_dir> \
    --scale 528 \
    --cache-dir data/cache/test_swinir_dn15 \
    --suffix effb7_528_tta90_hflip_dn15cache \
    --apply-ts

Notes:
  - Uses create_submission.py under the hood (cache-first loading).
  - If --apply-ts is set, tries to fit temperature on validation split defined by config.split.predefined_split.
    Falls back gracefully if split cannot be resolved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pickle

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.factory import create_model
from src.transforms.factory import create_transforms
from src.data.dataset import DocumentDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--scale', type=int, default=528)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--cache-dir', type=str, default=None)
    ap.add_argument('--suffix', type=str, default=None)
    ap.add_argument('--apply-ts', action='store_true')
    return ap.parse_args()


def build_temp_config(base_cfg: dict, scale: int, bs: int) -> Path:
    # Start from full base config to preserve model/paths/augmentations
    tmp = yaml.safe_load(yaml.safe_dump(base_cfg))
    tmp.setdefault('data', {})
    tmp['data']['input_size'] = scale
    tmp['data'].setdefault('mean', base_cfg['data'].get('mean', [0.485,0.456,0.406]))
    tmp['data'].setdefault('std', base_cfg['data'].get('std', [0.229,0.224,0.225]))
    ld = tmp['data'].setdefault('loader', {})
    ld['batch_size'] = bs
    ld['num_workers'] = base_cfg['data']['loader'].get('num_workers', 8)
    ld['pin_memory'] = True
    ld['persistent_workers'] = True
    ld['prefetch_factor'] = base_cfg['data']['loader'].get('prefetch_factor', 2)
    out = ROOT / 'configs' / f'_auto_tta_{scale}.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        yaml.safe_dump(tmp, f)
    return out


@torch.no_grad()
def fit_temperature(run_dir: Path, cfg: dict) -> float:
    """Fit temperature on validation split via NLL minimization.
    If split cannot be resolved, returns 1.0 (no scaling).
    """
    try:
        split_pkl = Path(cfg['split']['predefined_split'])
        with open(split_pkl, 'rb') as f:
            folds = pickle.load(f)
        fold_idx = int(cfg['split']['fold_index'])
        val_idx = None
        # Support common structures
        if isinstance(folds, dict):
            # e.g., {0: {'train': [...], 'val': [...]}, ...}
            if fold_idx in folds:
                block = folds[fold_idx]
                val_idx = block.get('val') or block.get('valid') or block.get('val_idx')
        elif isinstance(folds, list):
            block = folds[fold_idx]
            if isinstance(block, dict):
                val_idx = block.get('val') or block.get('valid') or block.get('val_idx')
            elif isinstance(block, (list, tuple)) and len(block) >= 2:
                # Common pattern: (train_idx, val_idx)
                val_idx = block[1]
        if val_idx is None:
            return 1.0

        train_csv = Path(cfg['paths']['train_csv'])
        df = pd.read_csv(train_csv)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(cfg).to(device)
        ckpt = run_dir / 'checkpoints' / 'best.pth'
        state = torch.load(ckpt, map_location=device)
        model_state = state.get('ema_state') or state.get('swa_state') or state.get('model_state') or state
        model.load_state_dict(model_state, strict=False)
        model.eval()

        tfm = create_transforms(cfg, is_train=False)
        ds = DocumentDataset(val_df, Path(cfg['paths']['image_dir']), transforms=tfm, is_train=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg['data']['loader'].get('batch_size', 8),
                                             shuffle=False, num_workers=cfg['data']['loader'].get('num_workers', 4),
                                             pin_memory=True)

        logits_list, labels_list = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            logits_list.append(out.detach().cpu())
            labels_list.append(y.detach().cpu())
        logits = torch.cat(logits_list, 0)
        labels = torch.cat(labels_list, 0)

        # Temperature scaling
        T = torch.ones(1, requires_grad=True, device='cpu')
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
        y = labels.long()

        def nll():
            opt.zero_grad()
            scaled = logits / T.clamp_min(1e-3)
            loss = torch.nn.functional.cross_entropy(scaled, y)
            loss.backward()
            return loss

        opt.step(nll)
        temp = float(T.detach().clamp_min(1e-3).item())
        # Save TS record
        ts_dir = ROOT / 'reports' / 'run_history'
        ts_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            'run_dir': str(run_dir), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'temp': temp, 'val_size': int(len(labels)),
        }
        with open(ts_dir / f"TS_{run_dir.name}.json", 'w') as f:
            json.dump(rec, f, indent=2)
        return temp
    except Exception:
        return 1.0


def apply_temperature_to_csv(csv_path: Path, probs_path: Path, temp: float):
    arr = np.load(probs_path)
    eps = 1e-12
    # convert probs to logits via log(p)
    logits = np.log(np.clip(arr, eps, 1.0))
    scaled = logits / max(temp, 1e-3)
    # softmax
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    df = pd.read_csv(csv_path)
    df['target'] = preds
    out = csv_path.with_name(csv_path.stem + f"_ts{temp:.3f}" + csv_path.suffix)
    df.to_csv(out, index=False)
    return out


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / 'config.yaml'
    cfg = yaml.safe_load(open(cfg_path))

    # Build temp config for scale/batch override
    tmp_cfg = build_temp_config(cfg, args.scale, args.batch_size)

    # Create submission with cache
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'create_submission.py'),
        '--config', str(tmp_cfg),
        '--checkpoint', str(run_dir / 'checkpoints' / 'best.pth'),
        '--suffix', args.suffix or f'auto_tta_cache_{args.scale}px',
        '--save-probs',
    ]
    if args.cache_dir:
        cmd.extend(['--denoise-cache-dir', args.cache_dir])
    subprocess.run(cmd, check=True)

    # Find the generated outputs
    subs_dir = ROOT / 'outputs' / 'submissions'
    latest = max(subs_dir.glob(f"{run_dir.name}_*.csv"), key=lambda p: p.stat().st_mtime)
    logits_dir = ROOT / 'outputs' / 'logits'
    probs = max(logits_dir.glob(latest.stem + '*.npy'), key=lambda p: p.stat().st_mtime)

    # Optional TS
    if args.apply_ts:
        temp = fit_temperature(run_dir, cfg)
        if temp != 1.0:
            out = apply_temperature_to_csv(latest, probs, temp)
            print(f"Temperature scaling applied: T={temp:.3f}\nScaled submission: {out}")
        else:
            print("Temperature scaling skipped or T=1.0")

    # Minimal report update
    rep_dir = ROOT / 'reports' / 'summary' / time.strftime('%Y%m%d-%H%M%S')
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / 'auto_tta_note.md', 'w') as f:
        f.write(f"Run: {run_dir.name}\nScale: {args.scale}\nCache: {args.cache_dir}\nSubmission: {latest.name}\n")

    print(f"Done. Submission: {latest}")


if __name__ == '__main__':
    main()
