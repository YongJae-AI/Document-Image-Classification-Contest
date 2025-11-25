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
    ap.add_argument('--scales', type=str, default=None, help='Comma-separated scales (e.g., "384,448,528"). If set, runs all scales and averages probs into one combined submission.')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--cache-dir', type=str, default=None)
    ap.add_argument('--denoise', type=str, default=None, choices=[None, 'swinir_proxy', 'swinir'])
    ap.add_argument('--suffix', type=str, default=None)
    ap.add_argument('--apply-ts', action='store_true')
    ap.add_argument('--temp-fixed', type=float, default=None, help='If set, skip fitting and apply this temperature directly.')
    # TTA overrides
    ap.add_argument('--angles', type=str, default=None,
                    help='Comma-separated rotation angles (e.g., "0,90,180,270" or "0,15,30,45,60"). If omitted, use config.')
    ap.add_argument('--hflip-only', action='store_true', help='Force HorizontalFlip on and VerticalFlip off (transpose not used).')
    ap.add_argument('--tile-grid', type=str, default=None, help='Enable tiled inference with grid, e.g., 3x3')
    ap.add_argument('--tile-overlap', type=float, default=0.125, help='Tile overlap ratio per axis (0.0~0.4)')
    ap.add_argument('--save-tile-probs', action='store_true', help='When tiling, save per-tile averaged probs to logits_tiles/*.npz')
    return ap.parse_args()


def build_temp_config(base_cfg: dict, scale: int, bs: int, angles: str | None = None, hflip_only: bool = False) -> Path:
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

    # TTA overrides
    sub = tmp.setdefault('submission', {})
    tta = sub.setdefault('tta', {})
    if angles is not None:
        try:
            rot = [int(a.strip()) for a in angles.split(',') if a.strip()!='']
        except Exception:
            rot = None
        if rot:
            tta['enabled'] = True
            tta['rotations'] = rot
            # disable jitter if explicit angles provided
            tta['jitter_degrees'] = 0
    if hflip_only:
        tta['enabled'] = True
        tta['horizontal_flip'] = True
        tta['vertical_flip'] = False

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
        # Case 1: predefined split available (CV 모드)
        val_df = None
        split_cfg = cfg.get('split', {})
        split_pkl = Path(split_cfg.get('predefined_split') or '')
        if split_pkl.exists():
            with open(split_pkl, 'rb') as f:
                folds = pickle.load(f)
            fold_idx = int(split_cfg.get('fold_index', 0))
            val_idx = None
            if isinstance(folds, dict):
                if fold_idx in folds:
                    block = folds[fold_idx]
                    val_idx = block.get('val') or block.get('valid') or block.get('val_idx')
            elif isinstance(folds, list):
                block = folds[fold_idx]
                if isinstance(block, dict):
                    val_idx = block.get('val') or block.get('valid') or block.get('val_idx')
                elif isinstance(block, (list, tuple)) and len(block) >= 2:
                    val_idx = block[1]
            if val_idx is not None:
                train_csv = Path(cfg['paths']['train_csv'])
                df = pd.read_csv(train_csv)
                val_df = df.iloc[val_idx].reset_index(drop=True)

        # Case 2: full_train 모드의 소홀드아웃 사용
        if val_df is None and bool(split_cfg.get('full_train', False)):
            holdout_frac = float(split_cfg.get('full_train_holdout_fraction', 0.0) or 0.0)
            if holdout_frac > 0.0:
                from sklearn.model_selection import StratifiedShuffleSplit
                seed = int(split_cfg.get('seed', 2024))
                df = pd.read_csv(Path(cfg['paths']['train_csv']))
                sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
                (_, val_idx), = sss.split(df['ID'], df['target'])
                val_df = df.iloc[val_idx].reset_index(drop=True)
        if val_df is None:
            return 1.0

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

    combined_probs: np.ndarray | None = None
    combined_stem: str | None = None
    used_scales = []

    def _run_one_scale(scale: int) -> tuple[Path, Path]:
        nonlocal combined_stem
        tmp_cfg = build_temp_config(cfg, scale, args.batch_size, angles=args.angles, hflip_only=args.hflip_only)
        cmd = [
            sys.executable, str(ROOT / 'scripts' / 'create_submission.py'),
            '--config', str(tmp_cfg),
            '--checkpoint', str(run_dir / 'checkpoints' / 'best.pth'),
            '--suffix', (args.suffix or f'auto_tta') + f'_{scale}px',
            '--save-probs',
        ]
        if args.cache_dir:
            cmd.extend(['--denoise-cache-dir', args.cache_dir])
        if args.denoise:
            cmd.extend(['--denoise', args.denoise])
        if args.tile_grid:
            cmd.extend(['--tile-grid', args.tile_grid, '--tile-overlap', str(args.tile_overlap)])
            if args.save_tile_probs:
                cmd.append('--save-tile-probs')
        subprocess.run(cmd, check=True)

        subs_dir = ROOT / 'outputs' / 'submissions'
        cands = list(subs_dir.glob(f"{run_dir.name}_*.csv")) or list(subs_dir.glob('*.csv'))
        if not cands:
            raise RuntimeError('No submission CSVs found in outputs/submissions')
        latest = max(cands, key=lambda p: p.stat().st_mtime)
        logits_dir = ROOT / 'outputs' / 'logits'
        prob_cands = list(logits_dir.glob(latest.stem + '*.npy')) or list(logits_dir.glob('*.npy'))
        if not prob_cands:
            raise RuntimeError('No logits NPY found corresponding to submission')
        probs_path = max(prob_cands, key=lambda p: p.stat().st_mtime)
        if combined_stem is None:
            combined_stem = latest.stem
        return latest, probs_path

    # Run either single scale or multiple and average
    if args.scales:
        scales = [int(s.strip()) for s in args.scales.split(',') if s.strip()]
        prob_arrays = []
        latest_csvs = []
        for sc in scales:
            latest_csv, probs_path = _run_one_scale(sc)
            latest_csvs.append(latest_csv)
            used_scales.append(sc)
            prob_arrays.append(np.load(probs_path))
        # Average probabilities
        combined_probs = np.mean(np.stack(prob_arrays, axis=0), axis=0)
        # Write combined CSV
        sample_path = ROOT / 'data' / 'raw' / 'sample_submission.csv'
        df = pd.read_csv(sample_path)
        preds = combined_probs.argmax(axis=1)
        df['target'] = preds
        subs_dir = ROOT / 'outputs' / 'submissions'
        subs_dir.mkdir(parents=True, exist_ok=True)
        tag = args.suffix or 'ms'
        out_csv = subs_dir / f"{run_dir.name}_ms{','.join(map(str,used_scales))}_{tag}.csv"
        df.to_csv(out_csv, index=False)
        # Save combined probs
        logits_dir = ROOT / 'outputs' / 'logits'
        logits_dir.mkdir(parents=True, exist_ok=True)
        np.save(logits_dir / (out_csv.stem + '.npy'), combined_probs)
        print(f"Multi-scale combined submission: {out_csv}")
    else:
        # Single scale path
        _run_one_scale(args.scale)

    # Optional TS
    if (args.apply_ts or args.temp_fixed is not None) and not args.scales:
        temp = args.temp_fixed if args.temp_fixed is not None else fit_temperature(run_dir, cfg)
        if temp != 1.0:
            out = apply_temperature_to_csv(latest, probs, temp)
            print(f"Temperature scaling applied: T={temp:.3f}\nScaled submission: {out}")
        else:
            print("Temperature scaling skipped or T=1.0")

    # Minimal report update
    rep_dir = ROOT / 'reports' / 'summary' / time.strftime('%Y%m%d-%H%M%S')
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / 'auto_tta_note.md', 'w') as f:
        if args.scales:
            f.write(f"Run: {run_dir.name}\nScales: {','.join(map(str,used_scales))}\nCache: {args.cache_dir}\nSubmission: multi-scale combined\n")
        else:
            f.write(f"Run: {run_dir.name}\nScale: {args.scale}\nCache: {args.cache_dir}\nSubmission: {latest.name}\n")

    if args.scales:
        print("Done. Multi-scale combined submission created.")
    else:
        print(f"Done. Submission: {latest}")


if __name__ == '__main__':
    main()
