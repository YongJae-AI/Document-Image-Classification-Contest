#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    ap = argparse.ArgumentParser(description='Generate per-class ensemble weights from run per_class stats')
    ap.add_argument('--out-dir', type=str, default=None, help='Output directory for JSONs (default reports/summary/<ts>/)')
    ap.add_argument('--runs', nargs='*', default=None, help='Explicit run dirs for models in order: effnet_b7 effnetv2_l convnextv2 swin_large')
    ap.add_argument('--prefer', type=str, choices=['cv','fulltrain'], default='cv', help='Prefer CV runs (default) or full-train runs when auto-detecting')
    ap.add_argument('--fold', type=int, default=0, help='When prefer=cv, try to select this fold index')
    return ap.parse_args()


def autodetect_runs(root: Path, prefer: str = 'cv', fold: int = 0) -> Dict[str, Path]:
    key_map = {
        'effnet_b7': 'tf_efficientnet_b7_ns',
        'effnetv2_l': 'tf_efficientnetv2_l',
        'convnextv2': 'convnextv2_large',
        'swin_large': 'swin_large_patch4_window12',
    }
    out = {}
    runs_dir = root / 'outputs' / 'runs'
    import yaml
    for k, pat in key_map.items():
        cands = sorted(runs_dir.glob(f'*{pat}*'), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f'No runs found for {k} ({pat})')
        chosen = None
        # Examine configs to filter by prefer
        cv_pool = []
        ft_pool = []
        for rd in reversed(cands):  # newest first
            cfgp = rd / 'config.yaml'
            if not cfgp.exists():
                continue
            try:
                cfg = yaml.safe_load(cfgp.read_text())
            except Exception:
                continue
            split = cfg.get('split', {})
            if split.get('full_train', False):
                ft_pool.append(rd)
            else:
                # CV run: optionally check fold
                if split.get('fold_index') == fold:
                    cv_pool.append(rd)
                else:
                    cv_pool.append(rd)
        if prefer == 'cv' and cv_pool:
            chosen = cv_pool[0]
        elif prefer == 'fulltrain' and ft_pool:
            chosen = ft_pool[0]
        else:
            # fallback to newest candidate
            chosen = cands[-1]
        out[k] = chosen
    return out


def load_per_class(rd: Path):
    import json as js
    for fn in ['per_class_best.json','per_class_latest.json']:
        p = rd / fn
        if p.exists():
            return js.loads(p.read_text())
    # attempt backfill if missing
    try:
        subprocess.run(['python','-u','scripts/backfill_per_class.py','--run-dir',str(rd),'--batch-size','16'], check=True)
        for fn in ['per_class_best.json','per_class_latest.json']:
            p = rd / fn
            if p.exists():
                return js.loads(p.read_text())
    except Exception:
        pass
    raise SystemExit(f'per_class json missing in {rd} (backfill failed)')


def compute_baseline_weights(accs: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    # accs: model -> [acc per class]
    n = len(next(iter(accs.values())))
    weights: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        items = sorted(((accs[m][i], m) for m in accs), reverse=True)
        top, m1 = items[0]
        second, m2 = items[1]
        margin = top - second
        if margin >= 0.02:
            base = {m1: 0.6, m2: 0.3}
        elif margin >= 0.005:
            base = {m1: 0.5, m2: 0.3}
        else:
            base = {m1: 0.45, m2: 0.35}
        # Distribute remainder to others
        rem = 1.0 - sum(base.values())
        for _, m in items[2:]:
            base[m] = rem / max(1, len(items) - 2)
        weights[str(i)] = base
    return weights


def make_variant(weights: Dict[str, Dict[str, float]], variant: str) -> Dict[str, Dict[str, float]]:
    import copy
    W = copy.deepcopy(weights)
    for cls, m2w in W.items():
        # order models by weight desc
        items = sorted(((w, m) for m, w in m2w.items()), reverse=True)
        if variant == 'boost_top':
            delta = 0.05
            top_w, top_m = items[0]
            new_top = min(0.99, top_w + delta)
            scale = (1.0 - new_top) / (1.0 - top_w) if (1.0 - top_w) > 1e-6 else 1.0
            m2w[top_m] = new_top
            for _, m in items[1:]:
                m2w[m] *= scale
        elif variant == 'boost_second':
            delta = 0.05
            sec_w, sec_m = items[1]
            new_sec = min(0.99, sec_w + delta)
            scale = (1.0 - new_sec) / (1.0 - sec_w) if (1.0 - sec_w) > 1e-6 else 1.0
            m2w[sec_m] = new_sec
            for _, m in items:
                if m == sec_m:
                    continue
                m2w[m] *= scale
        elif variant == 'conservative':
            # Drag towards uniform slightly
            k = len(m2w)
            for m in m2w:
                m2w[m] = 0.8 * m2w[m] + 0.2 / k
        elif variant == 'aggressive_top':
            # Push more mass to the top
            top_w, top_m = items[0]
            new_top = min(0.99, max(top_w, 0.6) + 0.05)
            scale = (1.0 - new_top) / (1.0 - top_w) if (1.0 - top_w) > 1e-6 else 1.0
            m2w[top_m] = new_top
            for _, m in items[1:]:
                m2w[m] *= scale
        # else: unchanged
    return W


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parent.parent
    runs_map = autodetect_runs(ROOT, prefer=args.prefer, fold=args.fold) if not args.runs else {
        'effnet_b7': Path(args.runs[0]),
        'effnetv2_l': Path(args.runs[1]),
        'convnextv2': Path(args.runs[2]),
        'swin_large': Path(args.runs[3]),
    }
    accs: Dict[str, List[float]] = {}
    for k, rd in runs_map.items():
        pc = load_per_class(rd)
        cor = pc['per_class_correct']; tot = pc['per_class_total']
        accs[k] = [(c/t if t else 0.0) for c, t in zip(cor, tot)]

    base = compute_baseline_weights(accs)
    ts = __import__('time').strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / 'reports' / 'summary' / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ensemble_weights_fold0.json').write_text(json.dumps(base, indent=2))
    for v in ['boost_top','boost_second','conservative','aggressive_top']:
        var = make_variant(base, v)
        (out_dir / f'ensemble_weights_fold0_{v}.json').write_text(json.dumps(var, indent=2))
    print('Saved weight sets to', out_dir)


if __name__ == '__main__':
    main()
