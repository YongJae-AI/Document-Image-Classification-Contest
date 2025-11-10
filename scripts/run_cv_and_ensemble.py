#!/usr/bin/env python
import argparse
import os
import re
import subprocess
import time
from pathlib import Path
import json
import pandas as pd
import yaml
import numpy as np
from scripts.auto_tta_submit import fit_temperature, apply_temperature_to_csv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=[
        'efficientnet_b7_offline_448:configs/efficientnet_b7_offline_448.yaml:effnet',
        'convnextv2_large_offline:configs/convnextv2_large_offline.yaml:other',
        'swin_large_patch4_window12_384_in22k_offline:configs/swin_in22k_offline.yaml:other',
        'efficientnetv2_l_offline:configs/efficientnetv2_l_offline.yaml:other',
    ], help='name:config:kind (kind=effnet|other)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--scale_main', type=int, default=528)
    ap.add_argument('--scale_aux', type=int, default=456)
    ap.add_argument('--infer_bs', type=int, default=2)
    ap.add_argument('--output', type=str, default='reports/summary/cv_ensemble_summary.json')
    return ap.parse_args()


def read_best_f1(run_dir: Path) -> float:
    log = run_dir / 'train.log'
    f1 = None
    if log.exists():
        for line in log.read_text().splitlines()[::-1]:
            m = re.search(r'Best f1: ([0-9.]+)', line)
            if m:
                f1 = float(m.group(1))
                break
    return f1 or 0.0


def main():
    args = parse_args()
    rows = []
    for spec in args.models:
        name, cfg, kind = spec.split(':')
        for fold in range(args.folds):
            # Train
            subprocess.run([
                'python', 'scripts/run_experiment.py', '--config', cfg, '--fold', str(fold)
            ], check=True)
            # Find latest run dir for this config+fold
            runs = sorted(Path('outputs/runs').glob(f'*_{name.split("_")[0]}*'), key=lambda p: p.stat().st_mtime)
            run_dir = runs[-1]
            f1 = read_best_f1(run_dir)
            rows.append({'model': name, 'fold': fold, 'best_f1': f1, 'run_dir': str(run_dir)})
            # Submissions for main scale (+ TS)
            if kind == 'effnet':
                # Build temp config overriding input_size / loader.batch_size
                base_cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
                base_cfg['data']['input_size'] = args.scale_main
                base_cfg['data']['loader']['batch_size'] = args.infer_bs
                tmp_cfg = Path('configs') / f'_cvtmp_{name}_{args.scale_main}.yaml'
                tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_cfg, 'w') as f:
                    yaml.safe_dump(base_cfg, f)
                # Create submission with OpenCV proxy denoise
                sub_suffix = f'{name}_proxy_{args.scale_main}'
                subprocess.run([
                    'python','scripts/create_submission.py','--config',str(tmp_cfg),
                    '--checkpoint', str(run_dir / 'checkpoints' / 'best.pth'),
                    '--denoise','swinir_proxy','--suffix', sub_suffix, '--save-probs'
                ], check=True)
                # TS fit/apply
                temp = fit_temperature(run_dir, base_cfg)
                subs_dir = Path('outputs/submissions')
                latest = max(subs_dir.glob(f"{run_dir.name}_*{sub_suffix}.csv"), key=lambda p: p.stat().st_mtime)
                probs = max(Path('outputs/logits').glob(latest.stem + '*.npy'), key=lambda p: p.stat().st_mtime)
                apply_temperature_to_csv(latest, probs, temp)
                # aux scale 456
                base_cfg['data']['input_size'] = args.scale_aux
                base_cfg['data']['loader']['batch_size'] = args.infer_bs
                tmp_cfg2 = Path('configs') / f'_cvtmp_{name}_{args.scale_aux}.yaml'
                with open(tmp_cfg2, 'w') as f:
                    yaml.safe_dump(base_cfg, f)
                sub_suffix2 = f'{name}_proxy_{args.scale_aux}'
                subprocess.run([
                    'python','scripts/create_submission.py','--config',str(tmp_cfg2),
                    '--checkpoint', str(run_dir / 'checkpoints' / 'best.pth'),
                    '--denoise','swinir_proxy','--suffix', sub_suffix2, '--save-probs'
                ], check=True)
                temp2 = fit_temperature(run_dir, base_cfg)
                latest2 = max(subs_dir.glob(f"{run_dir.name}_*{sub_suffix2}.csv"), key=lambda p: p.stat().st_mtime)
                probs2 = max(Path('outputs/logits').glob(latest2.stem + '*.npy'), key=lambda p: p.stat().st_mtime)
                apply_temperature_to_csv(latest2, probs2, temp2)
            else:
                # Non-effnet (no denoise), reuse auto_tta_submit to handle TS
                subprocess.run([
                    'python','scripts/auto_tta_submit.py','--run-dir',str(run_dir),
                    '--scale','384','--batch-size',str(args.infer_bs),'--apply-ts',
                    '--suffix', f'{name}_base384'
                ], check=True)

    # Save summary
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    # Also write CSV
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix('.csv'), index=False)
    print('Summary saved to', out, 'and', out.with_suffix('.csv'))


if __name__ == '__main__':
    main()
