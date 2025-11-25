#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def identify_model(run_name: str) -> str:
    n = run_name.lower()
    if 'tf_efficientnet_b7_ns' in n: return 'effnet_b7'
    if 'swin_large_patch4_window12' in n: return 'swin_large'
    if 'convnextv2' in n: return 'convnextv2'
    if 'efficientnetv2_l' in n: return 'effnetv2_l'
    return 'other'


def load_per_class(run_dir: Path) -> Dict | None:
    for fname in ['per_class_best.json','per_class_latest.json']:
        p = run_dir / fname
        if p.exists():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                return None
    return None


def main() -> None:
    rows: List[Dict] = []
    for rd in sorted((ROOT / 'outputs' / 'runs').glob('*')):
        if not rd.is_dir():
            continue
        model = identify_model(rd.name)
        if model == 'other':
            continue
        metrics = rd / 'metrics.jsonl'
        best_f1 = None
        last_f1 = None
        if metrics.exists():
            try:
                with metrics.open() as f:
                    for line in f:
                        j = json.loads(line)
                        if 'epoch' in j and 'f1' in j.get('metric', j):
                            pass
                # simple scan for last line
                last = json.loads(list(metrics.open())[-1])
                last_f1 = last.get('f1') or last.get('metric')
            except Exception:
                pass
        pc = load_per_class(rd)
        rows.append({
            'run_dir': rd.name,
            'model': model,
            'has_per_class': bool(pc),
            'last_metric': last_f1,
        })
    out = ROOT / 'experiments' / 'experiment_log_consolidated.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('Saved', out)


if __name__ == '__main__':
    main()

