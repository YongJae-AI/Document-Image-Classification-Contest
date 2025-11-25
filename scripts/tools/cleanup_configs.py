#!/usr/bin/env python
from __future__ import annotations
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / 'configs'

# Final targets (source -> dest filename)
FINAL_SOURCES = {
    'fulltrain_efficientnet_b7_offline_448.yaml': 'efficientnet_b7.yaml',
    'fulltrain_swin_in22k_offline.yaml': 'swin_large_384.yaml',
    'fulltrain_convnextv2_large_offline.yaml': 'convnextv2_large_384.yaml',
    'fulltrain_efficientnetv2_l_offline.yaml': 'efficientnetv2_l_384.yaml',
}

TEMPLATES = {
    'train_cv_template.yaml': 'efficientnet_b7_offline_448.yaml',
    'fulltrain_template.yaml': 'fulltrain_efficientnet_b7_offline_448.yaml',
}


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def archive_rest() -> None:
    ts = time.strftime('%Y%m%d-%H%M%S')
    arch = CFG / 'archive' / ts
    arch.mkdir(parents=True, exist_ok=True)
    # Move everything except final/ templates/ archive/ and selected finals
    keep_dirs = {'final', 'templates', 'archive'}
    keep_files = set(FINAL_SOURCES.keys()) | set(TEMPLATES.values())
    for p in CFG.iterdir():
        if p.name in keep_dirs:
            continue
        if p.is_file() and p.name in keep_files:
            continue
        # move
        shutil.move(str(p), str(arch / p.name))


def main():
    # 1) Copy finals
    for src_name, dst_name in FINAL_SOURCES.items():
        src = CFG / src_name
        if src.exists():
            safe_copy(src, CFG / 'final' / dst_name)
    # 2) Create templates from representative files
    for dst_name, src_name in TEMPLATES.items():
        src = CFG / src_name
        if src.exists():
            safe_copy(src, CFG / 'templates' / dst_name)
    # 3) Archive the rest
    archive_rest()
    print('Configs cleanup done. See configs/final, configs/templates, configs/archive')


if __name__ == '__main__':
    main()

