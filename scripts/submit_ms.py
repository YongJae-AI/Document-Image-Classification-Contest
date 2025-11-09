#!/usr/bin/env python
from __future__ import annotations
import sys, subprocess, yaml
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: submit_ms.py <run_dir> <size>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    size = int(sys.argv[2])
    cfg_in = run_dir / 'config.yaml'
    ckpt = run_dir / 'checkpoints' / 'best.pth'
    if not cfg_in.exists() or not ckpt.exists():
        print("Missing config or checkpoint", cfg_in, ckpt)
        sys.exit(2)
    cfg = yaml.safe_load(cfg_in.read_text())
    cfg['data']['input_size'] = size
    # reduce inference batch to avoid OOM during SwinIR + classifier TTA
    cfg['data']['loader']['batch_size'] = 2
    tmp_cfg = Path(f'configs/_tmp_ms_{size}.yaml')
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    swinir = '/root/SwinIR/experiments/pretrained_models/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth'
    suffix = f'effb7_ms{size}_swinir15'
    cmd = [sys.executable, 'scripts/create_submission.py',
           '--config', str(tmp_cfg), '--checkpoint', str(ckpt),
           '--denoise', 'swinir', '--swinir-weights', swinir,
           '--suffix', suffix, '--skip-if-unchanged']
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
