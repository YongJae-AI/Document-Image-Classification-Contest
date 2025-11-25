#!/usr/bin/env python
import argparse
from pathlib import Path
import re
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description='Create relabel visualization (grid + table) from markdown log')
    ap.add_argument('--log', default='reports/manual_review/relabel_20251107/relabel_log.md')
    ap.add_argument('--count', type=int, default=8)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--no-captions', action='store_true', help='그리드에 캡션(타이틀) 제거')
    return ap.parse_args()


def parse_log(md_path: Path):
    rows = []
    pat = re.compile(r"\|\s*([\w\.]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")
    for line in md_path.read_text(encoding='utf-8').splitlines():
        m = pat.search(line)
        if m:
            img, old, new = m.group(1), int(m.group(2)), int(m.group(3))
            rows.append({'id': img, 'old': old, 'new': new})
    return rows


def find_image_path(img_id: str, roots):
    for d in roots:
        p = d / img_id
        if p.exists():
            return p
    raise FileNotFoundError(img_id)


def imread_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    md = Path(args.log)
    rows = parse_log(md)
    if not rows:
        raise SystemExit('No entries parsed from log')
    rows = rows[: args.count]

    # outputs
    ts = time.strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else (Path('reports/summary') / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_tight' if args.no_captions else ''
    p_grid = out_dir / f'relabel_grid_{args.count}{suffix}.png'
    p_tbl_png = out_dir / f'relabel_table_{args.count}.png'
    p_tbl_csv = out_dir / f'relabel_table_{args.count}.csv'

    # image search dirs
    roots = [
        Path('data/processed/offline_aug/images'),
        Path('data/raw/train'),
        Path('data/raw/test'),
    ]

    # grid: 2x4 when 8, else best effort
    n = len(rows)
    if n == 8:
        R, C = 2, 4
    elif n == 6:
        R, C = 2, 3
    else:
        C = min(4, n)
        R = int(np.ceil(n / C))

    fig, axes = plt.subplots(R, C, figsize=(4*C, 3*R))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
    for ax, item in zip(axes, rows):
        p = find_image_path(item['id'], roots)
        img = imread_rgb(p)
        ax.imshow(img)
        if not args.no_captions:
            ax.set_title(f"{item['id']} ({item['old']} -> {item['new']})", fontsize=9)
    if args.no_captions:
        # Remove all paddings/whitespace for PPT embedding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    else:
        plt.tight_layout()
    plt.savefig(p_grid, dpi=150)
    plt.close()

    df = pd.DataFrame(rows)[['id','old','new']]
    df.to_csv(p_tbl_csv, index=False)
    fig, ax = plt.subplots(figsize=(min(12, 2 + 1.2*len(df)), 2.5))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(p_tbl_png, dpi=150)
    plt.close()

    print('Saved:', p_grid)
    print('Saved:', p_tbl_png)
    print('Saved:', p_tbl_csv)


if __name__ == '__main__':
    main()
