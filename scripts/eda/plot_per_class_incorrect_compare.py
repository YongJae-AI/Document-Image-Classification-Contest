#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load_per_class(run_dir: Path) -> Dict | None:
    for fname in ["per_class_best.json", "per_class_latest.json"]:
        p = run_dir / fname
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def latest_run_for_key(key_substr: str) -> Path | None:
    cands = sorted((ROOT / 'outputs' / 'runs').glob(f"*{key_substr}*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser(description='Compare per-class incorrect ratio across models in one figure')
    ap.add_argument('--runs', nargs='*', default=None, help='Run dir names under outputs/runs (exact). If omitted, auto-detect latest 4 backbones.')
    ap.add_argument('--labels', nargs='*', default=None, help='Legend labels for runs (same order as --runs).')
    ap.add_argument('--dest', type=str, default='reports/per_class', help='Output directory')
    ap.add_argument('--counts', action='store_true', help='Plot incorrect counts instead of ratios')
    ap.add_argument('--out-name', type=str, default=None, help='Output filename (png). If omitted, uses per_class_incorrect_compare_fold0.png')
    ap.add_argument('--title', type=str, default='Per Class Incorrect (Fold0) — Compare')
    args = ap.parse_args()

    if args.runs:
        runs: List[Tuple[str, Path]] = [(lab if args.labels and i < len(args.labels) else Path(r).name, ROOT / 'outputs' / 'runs' / r)
                                        for i, (lab, r) in enumerate(zip(args.labels or [], args.runs))]
        # If labels shorter, fill with run names
        if len(runs) == 0:
            runs = [(Path(r).name, ROOT / 'outputs' / 'runs' / r) for r in args.runs]
    else:
        # Auto-pick latest for 4 backbones
        keys = [
            ('EffNet-B7', 'tf_efficientnet_b7_ns'),
            ('Swin-L', 'swin_large_patch4_window12_384_in22k'),
            ('ConvNeXtV2-L', 'convnextv2_large'),
            ('EffNetV2-L', 'tf_efficientnetv2_l'),
        ]
        runs = []
        for lab, key in keys:
            rd = latest_run_for_key(key)
            if rd:
                runs.append((lab, rd))

    if not runs:
        print('No runs found')
        return

    # Load per-class stats
    series: List[Tuple[str, np.ndarray]] = []
    num_classes = None
    for lab, rd in runs:
        pc = load_per_class(rd)
        if not pc:
            print(f"[Skip] No per-class stats: {rd}")
            continue
        tot = np.array(pc.get('per_class_total', []), dtype=float)
        cor = np.array(pc.get('per_class_correct', []), dtype=float)
        if num_classes is None:
            num_classes = len(tot)
        if args.counts:
            inc = np.maximum(0.0, tot - cor)
        else:
            inc = np.where(tot > 0, (tot - cor) / tot, 0.0)
        series.append((lab, inc))

    if not series:
        print('No valid per-class stats to plot')
        return

    ncls = len(series[0][1])
    idx = np.arange(ncls)
    nmodel = len(series)
    width = max(0.1, min(0.8 / nmodel, 0.22))
    plt.figure(figsize=(max(12, ncls * 0.6), 6))

    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i, (lab, inc) in enumerate(series):
        plt.bar(idx + (i - (nmodel-1)/2) * width, inc, width, label=lab, color=palette[i % len(palette)])

    plt.xlabel('class id'); plt.ylabel('incorrect ratio')
    plt.title(args.title)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)
    # x ticks every class id
    plt.xticks(idx, [str(int(i)) for i in idx])
    plt.legend(ncol=min(nmodel, 4))
    plt.tight_layout()

    out_dir = ROOT / args.dest
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.out_name if args.out_name else 'per_class_incorrect_compare_fold0.png')
    plt.savefig(out_path, dpi=180)
    plt.close()
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
