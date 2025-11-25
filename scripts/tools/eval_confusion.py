#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.factory import create_model
from src.transforms.factory import create_transforms
from src.data.dataset import DocumentDataset


def load_cfg(run_dir: Path) -> Dict[str, Any]:
    return yaml.safe_load((run_dir / 'config.yaml').read_text())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dirs', nargs='+', required=True, help='One or more run directories under outputs/runs')
    ap.add_argument('--normalize', action='store_true')
    args = ap.parse_args()

    for rd_name in args.run_dirs:
        run_dir = (ROOT / 'outputs' / 'runs' / rd_name).resolve()
        if not run_dir.exists():
            print(f"[Skip] {rd_name}: not found")
            continue
        cfg = load_cfg(run_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset (validation fold)
        split = cfg.get('split', {})
        if split.get('full_train', False):
            print(f"[Warn] {rd_name}: full_train config — no validation fold available. Skipping.")
            continue
        import pickle
        with open(split.get('predefined_split'), 'rb') as f:
            folds = pickle.load(f)
        fi = int(split.get('fold_index', 0))
        if isinstance(folds, list):
            tr_idx, va_idx = folds[fi]
        elif isinstance(folds, dict):
            blk = folds[fi]
            va_idx = blk.get('val') or blk.get('valid') or blk.get('val_idx')
        else:
            print(f"[Skip] {rd_name}: unsupported split file")
            continue
        df = pd.read_csv(cfg['paths']['train_csv'])
        val_df = df.iloc[va_idx].reset_index(drop=True)
        tfm = create_transforms(cfg, is_train=False)
        ds = DocumentDataset(val_df, Path(cfg['paths']['image_dir']), transforms=tfm, is_train=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg['data']['loader'].get('batch_size', 8),
                                             shuffle=False, num_workers=cfg['data']['loader'].get('num_workers', 4),
                                             pin_memory=True)

        # Model
        model = create_model(cfg).to(device)
        state = torch.load(run_dir / 'checkpoints' / 'best.pth', map_location=device)
        mstate = state.get('ema_state') or state.get('swa_state') or state.get('model_state') or state
        model.load_state_dict(mstate, strict=False)
        model.eval()

        num_classes = int(cfg['data']['num_classes'])
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            if hasattr(out, 'logits'):
                out = out.logits
            pred = out.argmax(1)
            for ti, pi in zip(y.view(-1).tolist(), pred.view(-1).tolist()):
                cm[int(ti), int(pi)] += 1

        plots_dir = run_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        # save CSV
        pd.DataFrame(cm).to_csv(plots_dir / 'confusion_epochXX.csv', index=False)
        print(f"[Saved] {plots_dir / 'confusion_epochXX.csv'}")

        # optional annotated heatmaps
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            for norm in (False, True):
                mat = cm.astype(float)
                if norm:
                    rs = mat.sum(axis=1, keepdims=True); rs[rs==0]=1
                    mat = mat / rs
                plt.figure(figsize=(max(6, num_classes*0.5), max(5, num_classes*0.5)))
                ax = sns.heatmap(mat, cmap='Blues', annot=True, fmt='.2f' if norm else 'd', cbar=True, annot_kws={'fontsize':7})
                ax.set_xlabel('Predicted'); ax.set_ylabel('True')
                ax.set_title(f"Confusion — {run_dir.name}{' (Normalized)' if norm else ''}")
                plt.tight_layout()
                outp = plots_dir / f"confusion_epochXX{'_norm' if norm else ''}_annot.png"
                plt.savefig(outp, dpi=180); plt.close()
                print(f"[Saved] {outp}")
        except Exception:
            pass


if __name__ == '__main__':
    main()

