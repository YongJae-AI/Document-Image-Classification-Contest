#!/usr/bin/env python
"""
Pipeline to (1) train only the remaining models (default: ConvNeXtV2-L, EfficientNetV2-L),
(2) create submissions (TTA+TS) for all four models (EffNet-B7, Swin-L, ConvNeXtV2-L, EffNetV2-L),
and (3) run per-class weighted ensemble.

Usage examples:
  # Default: train convnextv2 & effnetv2_l, submit all, ensemble with multiple weight sets
  python -u scripts/posttrain_pipeline.py \
    --do-train --do-submit --do-ensemble --apply-ts-refit \
    --weights-json reports/summary/<ts>/ensemble_weights_fold0.json \
    --weights-json-multi reports/summary/<ts>/ensemble_weights_fold0_boost_top.json \
                         reports/summary/<ts>/ensemble_weights_fold0_boost_second.json \
                         reports/summary/<ts>/ensemble_weights_fold0_conservative.json \
                         reports/summary/<ts>/ensemble_weights_fold0_aggressive_top.json

  # If training is already done for the remaining models
  python -u scripts/posttrain_pipeline.py --do-submit --do-ensemble --apply-ts-refit \
    --weights-json reports/summary/<ts>/ensemble_weights_fold0.json
"""
import argparse
import json
import time
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


TRAIN_SPECS_DEFAULT = [
    # Only the remaining models to train by default
    {
        'alias': 'convnextv2',
        'config': 'configs/fulltrain_convnextv2_large_offline.yaml',
    },
    {
        'alias': 'effnetv2_l',
        'config': 'configs/fulltrain_efficientnetv2_l_offline.yaml',
    },
]

SUBMIT_SPECS = [
    # All four models for submission
    {
        'alias': 'effnet_b7',
        'key': 'tf_efficientnet_b7_ns',
        'scale': 528,
        'denoise': None,  # explicit OFF per user request
        'suffix': 'effb7_528_tta_no_denoise',
    },
    {
        'alias': 'swin_large',
        'key': 'swin_large_patch4_window12',
        'scale': 384,
        'denoise': None,
        'suffix': 'swinl_384_tta',
    },
    {
        'alias': 'convnextv2',
        'key': 'convnextv2_large',
        'scale': 384,
        'denoise': None,
        'suffix': 'convnextv2_384_tta',
    },
    {
        'alias': 'effnetv2_l',
        'key': 'tf_efficientnetv2_l',
        'scale': 384,
        'denoise': None,
        'suffix': 'effv2l_384_tta',
    },
]


def latest_run_dir_for_model_key(key: str) -> Path:
    runs = sorted((ROOT / 'outputs' / 'runs').glob(f"*{key}*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise RuntimeError(f'No run dir matches key: {key}')
    return runs[-1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--do-train', action='store_true')
    ap.add_argument('--do-submit', action='store_true')
    ap.add_argument('--do-ensemble', action='store_true')
    ap.add_argument('--apply-ts-refit', action='store_true', help='Refit temperature using holdout if available')
    ap.add_argument('--weights-json', type=str, default=None)
    ap.add_argument('--weights-json-multi', nargs='*', default=None)
    ap.add_argument('--weights-dir', type=str, default=None, help='Directory containing ensemble_weights_fold0*.json')
    ap.add_argument('--auto-weights', action='store_true', help='Auto-generate per-class weights via gen_perclass_weights.py and use them')
    # Advanced: customize which models to train
    ap.add_argument('--train-models', nargs='*', default=['convnextv2','effnetv2_l'],
                    help='Subset to train (aliases): convnextv2, effnetv2_l')
    return ap.parse_args()


def main():
    args = parse_args()
    # 1) Train only remaining models (sequential)
    if args.do_train:
        train_sel = set(getattr(args, 'train_models', ['convnextv2','effnetv2_l']))
        for sp in TRAIN_SPECS_DEFAULT:
            if sp['alias'] not in train_sel:
                continue
            cfg = ROOT / sp['config']
            print(f"[Train] {sp['alias']} using {cfg}")
            subprocess.run(['python','-u', str(ROOT / 'scripts' / 'run_experiment.py'), '--config', str(cfg)], check=True)

    probs = []
    # 2) Create submissions for all four models
    if args.do_submit:
        for sp in SUBMIT_SPECS:
            run_dir = latest_run_dir_for_model_key(sp['key'])
            print(f"[Submit] {sp['alias']} run={run_dir.name} scale={sp['scale']}")
            cmd = [
                'python','-u', str(ROOT / 'scripts' / 'auto_tta_submit.py'),
                '--run-dir', str(run_dir),
                '--scale', str(sp['scale']),
                '--batch-size', '2',
                '--suffix', sp['suffix'],
            ]
            if args.apply_ts_refit:
                cmd += ['--apply-ts']
            if sp['denoise']:
                cmd += ['--denoise', sp['denoise']]
            subprocess.run(cmd, check=True)
            # collect latest probs for ensemble
            subs_dir = ROOT / 'outputs' / 'submissions'
            # Prefer files starting with run_dir name; fallback to any latest CSV
            pref = list(subs_dir.glob(f"{run_dir.name}_*.csv"))
            if not pref:
                pref = list(subs_dir.glob('*.csv'))
            if not pref:
                raise RuntimeError('No submission CSVs found after submit step')
            latest_csv = max(pref, key=lambda p: p.stat().st_mtime)
            logits_dir = ROOT / 'outputs' / 'logits'
            cand_probs = sorted(logits_dir.glob(latest_csv.stem + '*.npy'), key=lambda p: p.stat().st_mtime)
            if not cand_probs:
                # Fallback to any latest npy
                cand_probs = sorted(logits_dir.glob('*.npy'), key=lambda p: p.stat().st_mtime)
            if cand_probs:
                probs.append(cand_probs[-1])

    # 3) Per-class ensemble
    if args.do_ensemble:
        # If probs are empty (e.g., user only requested --do-ensemble),
        # auto-discover latest logits per model to proceed without re-submitting.
        if len(probs) < 2:
            logits_dir = ROOT / 'outputs' / 'logits'
            if logits_dir.exists():
                model_keys = {
                    'effnet_b7': 'tf_efficientnet_b7_ns',
                    'swin_large': 'swin_large_patch4_window12',
                    'convnextv2': 'convnextv2_large',
                    'effnetv2_l': 'tf_efficientnetv2_l',
                }
                discovered = []
                for key in model_keys.values():
                    cands = sorted(logits_dir.glob(f"*{key}*.npy"), key=lambda p: p.stat().st_mtime)
                    if cands:
                        discovered.append(cands[-1])
                if len(discovered) >= 2:
                    probs = discovered
            # Still not enough
            if len(probs) < 2:
                raise RuntimeError('Need at least two probability files for ensemble (no submissions logits found).')
        ts = time.strftime('%Y%m%d-%H%M%S')
        weight_sets = []

        # Option A: auto-generate weights and use them
        weights_dir_path = None
        if args.auto_weights:
            print('[Ensemble] Auto-generating per-class weights...')
            auto_ok = False
            try:
                subprocess.run(['python','-u', str(ROOT / 'scripts' / 'gen_perclass_weights.py')], check=True)
                # pick latest reports/summary/* folder
                cand_dirs = sorted((ROOT / 'reports' / 'summary').glob('*'), key=lambda p: p.stat().st_mtime)
                if cand_dirs:
                    candidate = cand_dirs[-1]
                    # verify essential JSON exists
                    if (candidate / 'ensemble_weights_fold0.json').exists():
                        weights_dir_path = candidate
                        auto_ok = True
            except Exception as e:
                print('[Ensemble] Auto-generation failed, will try fallback to existing weights. Reason:', e)
            if not auto_ok:
                # Fallback: find latest directory that already contains ensemble_weights_fold0.json
                print('[Ensemble] Falling back to existing weights directory (Option B).')
                cand = None
                for d in sorted((ROOT / 'reports' / 'summary').glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
                    if (d / 'ensemble_weights_fold0.json').exists():
                        cand = d
                        break
                if cand is None:
                    raise RuntimeError('No existing weights directory found for fallback (expect ensemble_weights_fold0.json).')
                weights_dir_path = cand
        elif args.weights_dir:
            weights_dir_path = Path(args.weights_dir)
            # If variants are missing in the specified dir, generate them in-place
            needed = [
                'ensemble_weights_fold0.json',
                'ensemble_weights_fold0_boost_top.json',
                'ensemble_weights_fold0_boost_second.json',
                'ensemble_weights_fold0_conservative.json',
                'ensemble_weights_fold0_aggressive_top.json'
            ]
            missing = [n for n in needed if not (weights_dir_path / n).exists()]
            if missing:
                print(f"[Ensemble] Missing weight variants in {weights_dir_path}: {missing}. Generating variants...")
                # Prefer CV fold 0 by default to stay consistent
                subprocess.run([
                    'python','-u', str(ROOT / 'scripts' / 'gen_perclass_weights.py'),
                    '--prefer','cv','--fold','0','--out-dir', str(weights_dir_path)
                ], check=True)

        if weights_dir_path is not None:
            # collect known patterns from dir
            for name in ['ensemble_weights_fold0.json',
                         'ensemble_weights_fold0_boost_top.json',
                         'ensemble_weights_fold0_boost_second.json',
                         'ensemble_weights_fold0_conservative.json',
                         'ensemble_weights_fold0_aggressive_top.json']:
                p = weights_dir_path / name
                if p.exists():
                    weight_sets.append(str(p))

        # Option B: explicit lists
        if args.weights_json:
            weight_sets.append(args.weights_json)
        w_multi = args.weights_json_multi
        if w_multi:
            weight_sets.extend(w_multi)
        if not weight_sets:
            weight_sets = [None]
        for wj in weight_sets:
            suffix = Path(wj).stem if wj else 'uniform'
            out_csv = ROOT / 'outputs' / 'submissions' / f'{ts}_perclass_ensemble_{suffix}.csv'
            cmd = [
                'python','-u', str(ROOT / 'scripts' / 'ensemble_per_class.py'),
                '--sample-sub', 'data/raw/sample_submission.csv',
                '--probs', *[str(p) for p in probs],
                '--models', 'effnet_b7', 'effnetv2_l', 'convnextv2', 'swin_large',
                '--out-csv', str(out_csv)
            ]
            if wj:
                cmd += ['--weights-json', wj]
            subprocess.run(cmd, check=True)
            print('[Ensemble] Saved:', out_csv)


if __name__ == '__main__':
    main()
