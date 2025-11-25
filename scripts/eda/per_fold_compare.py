#!/usr/bin/env python
import argparse
import os, glob, json, time, subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def identify_model(run_name: str) -> str:
    n = run_name.lower()
    if 'tf_efficientnet_b7_ns' in n:
        return 'effnet_b7'
    if 'swin_large_patch4_window12' in n:
        return 'swin_large'
    if 'efficientnetv2_l' in n:
        return 'effnetv2_l'
    if 'convnextv2' in n:
        return 'convnextv2'
    return 'other'


def load_fold(run_dir: Path):
    try:
        cfg = yaml.safe_load((run_dir / 'config.yaml').read_text())
        return int(cfg['split']['fold_index'])
    except Exception:
        return None


def load_per_class(run_dir: Path):
    for name in ['per_class_best.json', 'per_class_latest.json']:
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def backfill_if_needed(run_dir: Path):
    pc = load_per_class(run_dir)
    if pc is not None:
        return True
    # try backfill
    try:
        subprocess.run([
            'python', 'scripts/backfill_per_class.py', '--run-dir', str(run_dir), '--batch-size', '16'
        ], check=True)
        return load_per_class(run_dir) is not None
    except Exception:
        return False


def collect_runs() -> pd.DataFrame:
    rows = []
    for p in sorted([q for q in glob.glob('outputs/runs/*') if os.path.isdir(q)]):
        rd = Path(p)
        model = identify_model(rd.name)
        if model == 'other':
            continue
        fold = load_fold(rd)
        rows.append({'run_dir': rd, 'model': model, 'fold': fold})
    return pd.DataFrame(rows)


def plot_overlay(fold: int, wanted: List[str]):
    df = collect_runs()
    df = df.dropna(subset=['fold'])
    df = df[df['fold'] == fold]
    if df.empty:
        print(f'No runs for fold {fold}.')
        return None
    # pick first run per model for this fold
    chosen: Dict[str, Path] = {}
    for m in wanted:
        sub = df[df['model'] == m]
        if sub.empty:
            continue
        rd = sub.iloc[0]['run_dir']
        if backfill_if_needed(rd):
            chosen[m] = rd
        else:
            print('skip (no per_class):', rd)
    if not chosen:
        print('No models with per_class available.')
        return None

    # load per_class and build series
    series: Dict[str, Dict[str, List[int]]] = {}
    n_cls = None
    for m, rd in chosen.items():
        pc = load_per_class(rd)
        if not pc:
            continue
        tot = pc.get('per_class_total') or []
        cor = pc.get('per_class_correct') or []
        if not tot or not cor:
            continue
        inc = [int(t) - int(c) for t, c in zip(tot, cor)]
        n_cls = len(tot)
        series[m] = {'total': tot, 'correct': cor, 'incorrect': inc}
    if not series or n_cls is None:
        print('Series empty after loading per_class.')
        return None

    # draw
    label_map = {
        'swin_large': 'Swin-L',
        'effnet_b7': 'EffNet-B7',
        'effnetv2_l': 'EffNetV2-L',
        'convnextv2': 'ConvNeXtV2'
    }
    colors = {
        'swin_large': '#1f78b4',
        'effnet_b7': '#33a02c',
        'effnetv2_l': '#e31a1c',
        'convnextv2': '#6a3d9a'
    }
    classes = np.arange(n_cls)
    width = 0.8 / max(1, len(series))
    out_dir = Path('reports/summary') / time.strftime('%Y%m%d-%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Correct overlay
    plt.figure(figsize=(12, 4))
    offs = -0.4
    for m in wanted:
        if m not in series:
            continue
        plt.bar(classes + offs, series[m]['correct'], width=width, label=label_map.get(m, m), color=colors.get(m, '#999999'))
        offs += width
    # X ticks every class id
    plt.xticks(classes, [str(int(i)) for i in classes])
    plt.title('Per Class Correct (Fold)')
    plt.xlabel('Class'); plt.ylabel('Count'); plt.legend(); plt.tight_layout()
    p1 = out_dir / f'per_fold_compare_fold{fold}_correct_overlay.png'
    plt.savefig(p1, dpi=150); plt.close()

    # Incorrect overlay
    plt.figure(figsize=(12, 4))
    offs = -0.4
    for m in wanted:
        if m not in series:
            continue
        plt.bar(classes + offs, series[m]['incorrect'], width=width, label=label_map.get(m, m), color=colors.get(m, '#999999'))
        offs += width
    # X ticks every class id
    plt.xticks(classes, [str(int(i)) for i in classes])
    plt.title('Per Class Incorrect (Fold)')
    plt.xlabel('Class'); plt.ylabel('Count'); plt.legend(); plt.tight_layout()
    p2 = out_dir / f'per_fold_compare_fold{fold}_incorrect_overlay.png'
    plt.savefig(p2, dpi=150); plt.close()

    print('Saved:', p1, p2)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description='Per-fold per-class overlay compare')
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--models', nargs='+', default=['effnet_b7', 'effnetv2_l', 'convnextv2'])
    args = ap.parse_args()
    out = plot_overlay(args.fold, args.models)
    if out:
        print('Artifacts at:', out)


if __name__ == '__main__':
    main()
