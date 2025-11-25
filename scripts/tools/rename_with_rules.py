#!/usr/bin/env python
"""
Repository-wide renaming helper to enforce naming rules:
 - Timestamps: YYYYMMDD-HHMMSS → MMDD-XX (per-day sequential index, 01..99)
 - Add minimal, human-readable qualifiers to certain JSON artifacts when possible
   (e.g., TS records: add T=<temp>; per-class ensemble: keep weight-set suffix).

Dry-run by default; use --apply to actually rename. Writes a CSV report of changes.

NOTE: This tool is conservative — it only renames files we are confident about
and skips anything ambiguous. You can rerun with --apply after reviewing the report.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

TS_PATTERN = re.compile(r"^(?:TS_)?(.+?)_ts\.json$")
STAMP_PATTERN = re.compile(r"(?P<ymd>20\d{6})-(?P<hms>\d{6})")


def find_candidates() -> List[Path]:
    exts = {'.json', '.csv', '.png', '.npy'}
    dirs = [ROOT / 'reports', ROOT / 'outputs']
    cands: List[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                if STAMP_PATTERN.search(p.name):
                    cands.append(p)
    return cands


def day_sequencer(paths: List[Path]) -> Dict[str, int]:
    """Assign per-day sequence numbers based on sorted mtime within the same day."""
    per_day: Dict[str, List[Tuple[float, Path]]] = {}
    for p in paths:
        m = STAMP_PATTERN.search(p.name)
        if not m:
            continue
        ymd = m.group('ymd')
        per_day.setdefault(ymd, []).append((p.stat().st_mtime, p))
    seqmap: Dict[str, int] = {}
    for ymd, items in per_day.items():
        for idx, (_, p) in enumerate(sorted(items), start=1):
            seqmap[p.as_posix()] = idx
    return seqmap


def qual_for_ts(path: Path) -> str | None:
    """If this is a TS json record we own, add T=<temp> in suffix."""
    try:
        js = json.loads(path.read_text())
        t = js.get('temp')
        if isinstance(t, (int, float)):
            return f"ts_T{float(t):.3f}"
    except Exception:
        pass
    return None


def build_new_name(p: Path, seqnum: int) -> str | None:
    m = STAMP_PATTERN.search(p.name)
    if not m:
        return None
    ymd = m.group('ymd')
    mmdd = ymd[4:]
    seq = f"{seqnum:02d}"
    # base without stamp
    base = STAMP_PATTERN.sub('', p.stem)
    base = base.strip('_-')
    # try to add qualifier for TS json
    qual = None
    if p.suffix.lower() == '.json':
        if p.parent.name == 'run_history' or 'ts' in p.stem:
            q = qual_for_ts(p)
            if q:
                qual = q
    parts = [mmdd, seq]
    if base:
        parts.append(base)
    if qual and qual not in base:
        parts.append(qual)
    return '_'.join(parts) + p.suffix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='Actually rename files')
    ap.add_argument('--report', default='rename_report.csv')
    args = ap.parse_args()

    cands = find_candidates()
    seqmap = day_sequencer(cands)
    changes: List[Tuple[str, str]] = []
    for p in cands:
        new = build_new_name(p, seqmap.get(p.as_posix(), 1))
        if not new:
            continue
        if new == p.name:
            continue
        target = p.with_name(new)
        changes.append((str(p), str(target)))

    # write report
    rep = ROOT / args.report
    with rep.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['old_path', 'new_path'])
        w.writerows(changes)
    print(f"Planned changes: {len(changes)} (report: {rep})")

    if args.apply:
        for old, new in changes:
            src = Path(old)
            dst = Path(new)
            if not dst.exists():
                src.rename(dst)
        print('Renaming applied.')
    else:
        print('Dry-run only. Re-run with --apply to rename.')


if __name__ == '__main__':
    main()

