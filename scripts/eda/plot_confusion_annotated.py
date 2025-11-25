#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent


def latest_confusion_csv(run_dir: Path) -> Optional[Path]:
    plots = run_dir / 'plots'
    if not plots.exists():
        return None
    cands = sorted(plots.glob('confusion_epoch*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def plot_confusion(csv_path: Path, out_dir: Path, title: str = '', normalize: bool = False) -> Path:
    df = pd.read_csv(csv_path)
    vals = df.values
    # try integer if looks like counts
    if np.issubdtype(vals.dtype, np.integer):
        cm = vals.astype(float)
        fmt_counts = True
    else:
        # some CSVs may be saved without dtype; try convert
        cm = vals.astype(float)
        # determine if all entries are whole numbers
        fmt_counts = np.allclose(cm, np.round(cm)) and not normalize
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    n = cm.shape[0]
    plt.figure(figsize=(max(6, n*0.5), max(5, n*0.5)))
    # ensure integer matrix when using 'd'
    if not normalize and fmt_counts:
        cm_disp = np.round(cm).astype(int)
        fmt = 'd'
    else:
        cm_disp = cm
        fmt = '.2f'
    ax = sns.heatmap(cm_disp, cmap='Blues', annot=True, fmt=fmt, cbar=True,
                     annot_kws={'fontsize': 7})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title or f'Confusion Matrix ({"normalized" if normalize else "counts"})')
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem + ('_norm' if normalize else '') + '_annot'
    out_path = out_dir / f'{stem}.png'
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None, help='Specific run dir names under outputs/runs. If omitted, scans all.')
    ap.add_argument('--dest', type=str, default='reports/confusion', help='Output directory for annotated confusion plots')
    ap.add_argument('--normalize', action='store_true', help='Normalize per true class (row-wise)')
    args = ap.parse_args()

    runs_root = ROOT / 'outputs' / 'runs'
    dest = ROOT / args.dest
    targets: List[Path]
    if args.runs:
        targets = [runs_root / r for r in args.runs]
    else:
        targets = sorted(runs_root.glob('*'))

    saved: List[Path] = []
    for rd in targets:
        if not rd.is_dir():
            continue
        csvp = latest_confusion_csv(rd)
        if not csvp:
            continue
        title = f'Confusion — {rd.name}'
        out1 = plot_confusion(csvp, dest, title=title, normalize=False)
        out2 = plot_confusion(csvp, dest, title=title + ' (Normalized)', normalize=True)
        saved.extend([out1, out2])

    if saved:
        print('Saved:')
        for p in saved:
            print(' -', p)
    else:
        print('No confusion CSVs found. Ensure training saved plots/confusion_epochXX.csv in run dirs.')


if __name__ == '__main__':
    main()
