#!/usr/bin/env python
import json, re, glob, time, os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_best_f1(log_path: Path):
    if not log_path.exists():
        return None
    best = None
    for line in log_path.read_text(errors='ignore').splitlines():
        m = re.search(r"Best f1: ([0-9\.]+)", line)
        if m:
            best = float(m.group(1))
    return best


def load_fold_index(cfg_path: Path):
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        return int(cfg['split']['fold_index'])
    except Exception:
        return None


def identify_model(run_name: str) -> str:
    name = run_name.lower()
    if 'tf_efficientnet_b7_ns' in name:
        return 'effnet_b7'
    if 'swin_large_patch4_window12' in name:
        return 'swin_large'
    if 'convnextv2' in name:
        return 'convnextv2'
    if 'efficientnetv2_l' in name:
        return 'effnetv2_l'
    return 'other'


def load_per_class(run_dir: Path):
    for fname in ['per_class_best.json', 'per_class_latest.json']:
        p = run_dir / fname
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def main():
    runs = sorted([p for p in glob.glob('outputs/runs/*') if os.path.isdir(p)])
    print('Found runs:', len(runs))
    if not runs:
        print('No runs found. Exiting.')
        return

    records = []
    for r in runs:
        rd = Path(r)
        model = identify_model(rd.name)
        if model == 'other':
            continue
        f1 = parse_best_f1(rd / 'train.log')
        fold = load_fold_index(rd / 'config.yaml')
        pc = load_per_class(rd)
        records.append({'run_dir': rd.name, 'model': model, 'fold': fold, 'best_f1': f1, 'per_class': pc})

    df = pd.DataFrame(records)
    if df.empty:
        print('No valid runs parsed. Exiting.')
        return
    df = df.dropna(subset=['fold', 'best_f1'])
    if df.empty:
        print('No runs with fold and best_f1. Exiting.')
        return
    df.sort_values(['model', 'fold'], inplace=True)

    out_dir = Path('reports/summary') / time.strftime('%Y%m%d-%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CV 성능/분산 요약
    summary = df.groupby('model')['best_f1'].agg(['mean', 'std', 'count']).reset_index()
    print('CV summary:\n', summary)
    if not summary.empty:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.bar(summary['model'], summary['mean'], yerr=summary['std'], capsize=4, color='#6aaed6', edgecolor='#333')
        ax.set_ylabel('CV F1 (mean ± sd)')
        ax.set_title('Model-wise CV performance')
        plt.tight_layout()
        plt.savefig(out_dir / 'cv_f1_by_model.png', dpi=150)
        plt.close()

    # 2) per-class accuracy 평균 히트맵
    rows = []
    for _, r in df.iterrows():
        pc = r['per_class'] or {}
        acc = pc.get('per_class_accuracy') or []
        if not acc:
            continue
        rows.append({'model': r['model'], 'fold': r['fold'], 'acc': acc})
    pcdf = pd.DataFrame(rows)
    mat: Dict[str, Any] = {}
    if not pcdf.empty:
        for model, gdf in pcdf.groupby('model'):
            arrs = [np.array(v, dtype=float) for v in gdf['acc']]
            arrs = [np.array([np.nan if (x is None) else x for x in a]) for a in arrs]
            try:
                mean_acc = np.nanmean(np.vstack(arrs), axis=0)
                mat[model] = mean_acc
            except Exception:
                pass
        if mat:
            models = list(mat.keys())
            classes = list(range(len(next(iter(mat.values())))))
            heat = pd.DataFrame({m: mat[m] for m in models}, index=classes)
            plt.figure(figsize=(8, 5))
            sns.heatmap(heat.T, cmap='Blues', vmin=0, vmax=1)
            plt.title('Per-class Accuracy (mean across folds)')
            plt.xlabel('Class'); plt.ylabel('Model')
            plt.tight_layout(); plt.savefig(out_dir / 'per_class_accuracy_heatmap.png', dpi=150); plt.close()

    # 3) 모델 보완성(effnet_b7 vs swin_large)
    if mat and 'effnet_b7' in mat and 'swin_large' in mat:
        A = mat['effnet_b7']; B = mat['swin_large']
        delta = A - B
        idx = np.argsort(delta)[::-1]
        xs = np.arange(len(delta))
        plt.figure(figsize=(10, 4))
        plt.bar(xs, delta[idx], color=['#1f78b4' if d >= 0 else '#e31a1c' for d in delta[idx]])
        plt.xticks(xs, [str(i) for i in idx], rotation=0)
        plt.axhline(0, color='#333', lw=1)
        plt.ylabel('Δ acc (effnet_b7 - swin_large)')
        plt.title('Complementarity by class (positive: effnet_b7 better)')
        plt.tight_layout(); plt.savefig(out_dir / 'complementarity_effb7_vs_swinl.png', dpi=150); plt.close()
        top5 = [(int(i), float(delta[i])) for i in idx[:5]]
        worst5 = [(int(i), float(delta[i])) for i in idx[-5:]]
        print('Top5 classes (effb7 > swinl):', top5)
        print('Worst5 classes (effb7 < swinl):', worst5)
    else:
        print('Complementarity plot skipped (missing models).')

    print('Artifacts saved to:', out_dir)


if __name__ == '__main__':
    main()

