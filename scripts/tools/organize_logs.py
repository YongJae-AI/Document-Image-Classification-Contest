#!/usr/bin/env python
from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Tuple
import shutil
import time

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / 'logs'


def map_name(name: str) -> str:
    base = name
    # common remaps to full words
    base = base.replace('effb7', 'efficientnet_b7')
    base = base.replace('swinl', 'swin_large')
    base = base.replace('convnextv2', 'convnextv2')  # already ok
    base = base.replace('submit_and_ensemble', 'submit_and_ensemble_pipeline')
    base = base.replace('ensemble_only', 'ensemble_perclass_weights')
    base = base.replace('pipeline_all', 'posttrain_pipeline_all')
    base = base.replace('offline_aug', 'offline_augmentation')
    base = base.replace('tta_submit', 'tta_submit')
    base = base.replace('run_b7_safe', 'run_b7_safe')
    # fold shorthand _fN -> _foldN
    base = re.sub(r'_f(\d)\b', r'_fold\1', base)
    return base


def unique_target(dst_dir: Path, name: str) -> Path:
    target = dst_dir / name
    if not target.exists():
        return target
    stem, suf = target.stem, target.suffix
    i = 2
    while True:
        cand = dst_dir / f"{stem}__{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(LOGS.glob('*')):
        if p.is_dir():
            continue
        stat = p.stat()
        ts = time.strftime('%Y%m%d', time.localtime(stat.st_mtime))
        day_dir = LOGS / ts
        day_dir.mkdir(parents=True, exist_ok=True)
        newname = map_name(p.name)
        target = unique_target(day_dir, newname)
        shutil.move(str(p), str(target))
        rows.append({
            'date': ts,
            'old_path': str(p.relative_to(ROOT)),
            'new_path': str(target.relative_to(ROOT)),
            'size_bytes': stat.st_size,
            'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
        })

    # write index
    idx = LOGS / 'logs_index.csv'
    with idx.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['date','old_path','new_path','size_bytes','mtime'])
        w.writeheader(); w.writerows(rows)
    print('Wrote', idx)


if __name__ == '__main__':
    main()

