#!/usr/bin/env python
"""
Train 4 models on full-train, create TTA+TS submissions, and build per-class weighted ensemble.

Usage:
  python -u scripts/fulltrain_pipeline.py \
    --do-train --do-submit --do-ensemble \
    --weights-json reports/summary/<ts>/ensemble_weights_fold0.json

Notes:
  - Training each model is sequential (OOM 안전)
  - TS는 fold 기록에 기반한 temp-fixed를 우선 사용(없으면 1.0)
  - EffNet-B7만 OpenCV denoise(swinir_proxy) 적용
  - TTA 추론 배치는 2로 고정
"""
import argparse
import json
import time
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent


SPECS = [
    {
        'alias': 'effnet_b7',
        'config': 'configs/fulltrain_efficientnet_b7_offline_448.yaml',
        'tta_scale': 528,
        'denoise': 'swinir_proxy',
        'temp_hint': ['*effb7*fold0*ts.json', '*effb7*cv_ts.json', '20251109-TS_effb7_448_fold0.json'],
    },
    {
        'alias': 'effnetv2_l',
        'config': 'configs/fulltrain_efficientnetv2_l_offline.yaml',
        'tta_scale': 384,
        'denoise': None,
        'temp_hint': ['*efficientnetv2l*fold0*ts.json'],
    },
    {
        'alias': 'convnextv2',
        'config': 'configs/fulltrain_convnextv2_large_offline.yaml',
        'tta_scale': 384,
        'denoise': None,
        'temp_hint': ['*convnextv2*fold0*ts.json'],
    },
    {
        'alias': 'swin_large',
        'config': 'configs/fulltrain_swin_in22k_offline.yaml',
        'tta_scale': 384,
        'denoise': None,
        'temp_hint': ['*swin_large*cv_ts.json', '*swin_large*fold*ts.json'],
    },
]


def latest_run_dir_for_model(model_key: str) -> Path:
    runs = sorted((ROOT / 'outputs' / 'runs').glob(f"*{model_key}*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise RuntimeError(f'No run dir matches {model_key}')
    return runs[-1]


def find_temp_from_history(hints) -> float:
    hist = ROOT / 'reports' / 'run_history'
    if not hist.exists():
        return 1.0
    cands = []
    for pat in hints:
        cands += list(hist.glob(pat))
    if not cands:
        return 1.0
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        rec = json.loads(cands[0].read_text())
        return float(rec.get('temperature') or rec.get('temp') or 1.0)
    except Exception:
        return 1.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--do-train', action='store_true')
    ap.add_argument('--do-submit', action='store_true')
    ap.add_argument('--do-ensemble', action='store_true')
    ap.add_argument('--weights-json', type=str, default=None)
    ap.add_argument('--weights-json-multi', nargs='*', default=None, help='Multiple weight jsons to try (runs ensemble for each)')
    ap.add_argument('--apply-ts-refit', action='store_true', help='Refit temperature using holdout if available')
    ap.add_argument('--no-denoise-b7', action='store_true', help='Disable denoise for EfficientNet-B7')
    return ap.parse_args()


def main():
    args = parse_args()
    subs = []
    probs = []

    for sp in SPECS:
        cfg = ROOT / sp['config']
        if args.do_train:
            print(f"[Train] {sp['alias']} using {cfg}")
            subprocess.run(['python','-u', str(ROOT / 'scripts' / 'run_experiment.py'), '--config', str(cfg)], check=True)
        if args.do_submit:
            key_map = {
                'effnet_b7': 'tf_efficientnet_b7_ns',
                'effnetv2_l': 'tf_efficientnetv2_l',
                'convnextv2': 'convnextv2_large',
                'swin_large': 'swin_large_patch4_window12',
            }
            run_dir = latest_run_dir_for_model(key_map[sp['alias']])
            temp = find_temp_from_history(sp['temp_hint'])
            print(f"[Submit] {sp['alias']} run={run_dir.name} scale={sp['tta_scale']} T={temp:.3f}")
            cmd = [
                'python','-u', str(ROOT / 'scripts' / 'auto_tta_submit.py'),
                '--run-dir', str(run_dir),
                '--scale', str(sp['tta_scale']),
                '--batch-size', '2',
                '--suffix', f"{sp['alias']}_{sp['tta_scale']}",
            ]
            if args.apply_ts_refit:
                cmd += ['--apply-ts']
            else:
                cmd += ['--temp-fixed', f"{temp:.6f}"]
            # Optional denoise for B7 only, unless disabled
            if sp['alias'] == 'effnet_b7' and not args.no_denoise_b7:
                cmd += ['--denoise', 'swinir_proxy']
            subprocess.run(cmd, check=True)
            # Locate latest submission and probs
            subs_dir = ROOT / 'outputs' / 'submissions'
            latest_csv = max(subs_dir.glob(f"{run_dir.name}_*.csv"), key=lambda p: p.stat().st_mtime)
            subs.append(latest_csv)
            logits_dir = ROOT / 'outputs' / 'logits'
            cand_probs = sorted(logits_dir.glob(latest_csv.stem + '*.npy'), key=lambda p: p.stat().st_mtime)
            if cand_probs:
                probs.append(cand_probs[-1])

    if args.do_ensemble:
        if len(probs) < 2:
            raise RuntimeError('Need at least two probability files for ensemble')
        ts = time.strftime('%Y%m%d-%H%M%S')
        weight_sets = []
        if args.weights_json:
            weight_sets.append(args.weights_json)
        if args.weights_json_multi:
            weight_sets.extend(args.weights_json_multi)
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
