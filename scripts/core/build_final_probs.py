#!/usr/bin/env python
"""
Build per-model final probability NPYs by combining multi-scale (448 highest weight)
and then combining type (tile vs non-split) with weights (tile:non = 0.6:0.4).

It searches outputs/logits for files produced by auto_tta_submit/create_submission
with the following suffix conventions:
  non-split: <...>_<TAG>_<SCALE>px.npy
  tiled    : <...>_<TAG>_tile33_<SCALE>px.npy

Model tags used:
  effnet_b7   -> TAG: effb7_ms_rot4_hflip
  swin_large  -> TAG: swinl_ms_rot4_hflip
  convnextv2  -> TAG: convnextv2_ms_rot4_hflip
  effnetv2_l  -> TAG: effv2l_ms_rot4_hflip

Outputs per model:
  outputs/logits/<timestamp>_<alias>_final.npy

Weights:
  - Multi-scale: prefer 448 highest. If (384,448,528) -> (0.3,0.4,0.3)
                  If (384,448)      -> (0.4,0.6)
                  If (single)       -> (1.0)
  - Type combine: tile:non = 0.6:0.4 when both exist; otherwise use the one available.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGITS = ROOT / 'outputs' / 'logits'

MODEL_SPECS = [
    ('effnet_b7',   'effb7_ms_rot4_hflip',      'tf_efficientnet_b7_ns'),
    ('swin_large',  'swinl_ms_rot4_hflip',      'swin_large_patch4_window12'),
    ('convnextv2',  'convnextv2_ms_rot4_hflip', 'convnextv2_large'),
    ('effnetv2_l',  'effv2l_ms_rot4_hflip',     'tf_efficientnetv2_l'),
]

def _latest_run_dir_for_key(key: str) -> Path | None:
    runs_root = ROOT / 'outputs' / 'runs'
    cands = sorted(runs_root.glob(f"*{key}*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None

def _load_ts_for_run(run_dir: Path) -> float | None:
    ts_dir = ROOT / 'reports' / 'run_history'
    js = ts_dir / f"TS_{run_dir.name}.json"
    if js.exists():
        import json
        try:
            rec = json.loads(js.read_text())
            return float(rec.get('temp', 1.0))
        except Exception:
            return None
    return None


def _find_scale_paths(tag: str, tiled: bool) -> dict[int, Path]:
    out: dict[int, Path] = {}
    pat = f"*{tag}*{'_tile33' if tiled else ''}_*px.npy"
    for p in LOGITS.glob(pat):
        name = p.stem
        # expect ..._<scale>px suffix
        try:
            scale_str = name.split('_')[-1]
            assert scale_str.endswith('px')
            sc = int(scale_str[:-2])
        except Exception:
            continue
        out[sc] = p
    return out


def _weighted_avg(arrs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    stack = np.stack(arrs, axis=0)
    # (K,N,C) * (K,) -> (N,C)
    return (stack * w[:, None, None]).sum(axis=0)


def _scale_weights(scales: list[int]) -> list[float]:
    # 448 highest, the rest equal
    sset = set(scales)
    if sset == {384, 448, 528}:
        return [0.3, 0.4, 0.3]  # order must match `scales`
    if sset == {384, 448}:
        # 448 highest
        return [0.4 if s==384 else 0.6 for s in scales]
    if sset == {448, 528}:
        # two scales but no 384: still 448 highest
        return [0.6 if s==448 else 0.4 for s in scales]
    # single-scale or other combos -> equal
    return [1.0 for _ in scales]


def _apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
    if temp is None or abs(temp - 1.0) < 1e-6:
        return probs
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))
    scaled = logits / max(temp, 1e-3)
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_one(alias: str, tag: str, model_key: str, tile_non_ratio=(0.6, 0.4), apply_ts: bool = False) -> Path | None:
    ns = _find_scale_paths(tag, tiled=False)
    ts = _find_scale_paths(tag, tiled=True)
    if not ns and not ts:
        return None
    # Non-split combine by scales
    ns_comb = None
    if ns:
        nscales = sorted(ns.keys())
        ns_arrays = [np.load(ns[s]) for s in nscales]
        ns_w = _scale_weights(nscales)
        ns_comb = _weighted_avg(ns_arrays, ns_w)
    # Tiled combine by scales
    ts_comb = None
    if ts:
        tscales = sorted(ts.keys())
        ts_arrays = [np.load(ts[s]) for s in tscales]
        ts_w = _scale_weights(tscales)
        ts_comb = _weighted_avg(ts_arrays, ts_w)
    # Type combine
    final: np.ndarray
    if ts_comb is not None and ns_comb is not None:
        a, b = tile_non_ratio
        final = a * ts_comb + b * ns_comb
    elif ts_comb is not None:
        final = ts_comb
    else:
        final = ns_comb  # type: ignore
    # Optional temperature scaling per model run
    if apply_ts:
        run_dir = _latest_run_dir_for_key(model_key)
        if run_dir is not None:
            T = _load_ts_for_run(run_dir)
            if T is not None:
                final = _apply_temperature(final, T)
                print(f"Applied TS (T={T:.3f}) to {alias}")

    out = LOGITS / f"{time.strftime('%Y%m%d-%H%M%S')}_{alias}_final.npy"
    LOGITS.mkdir(parents=True, exist_ok=True)
    np.save(out, final)
    print(f"Saved final probs for {alias}: {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile-weight', type=float, default=0.6)
    parser.add_argument('--non-weight', type=float, default=0.4)
    parser.add_argument('--apply-ts', action='store_true', help='Apply per-model temperature if available')
    args = parser.parse_args()
    paths = []
    for alias, tag, key in MODEL_SPECS:
        p = build_one(alias, tag, key, (args.tile_weight, args.non_weight), apply_ts=args.apply_ts)
        if p is not None:
            paths.append(p)
    if not paths:
        raise SystemExit('No probabilities found to combine.')
    print('Final model probs:')
    for p in paths:
        print(' -', p)

if __name__ == '__main__':
    main()
