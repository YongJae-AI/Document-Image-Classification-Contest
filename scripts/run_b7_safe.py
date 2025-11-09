#!/usr/bin/env python
from __future__ import annotations
import subprocess, sys, json, time
from pathlib import Path
import yaml

BASE_CFG = Path('configs/efficientnet_b7_offline_448.yaml')

def nvsmi_mem_used() -> int:
    try:
        out = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
        return int(out.splitlines()[0])
    except Exception:
        return 0

def patch_cfg(tmp: Path, batch: int, epochs: int, max_steps: int|None=None):
    data = yaml.safe_load(BASE_CFG.read_text())
    data['data']['loader']['batch_size'] = batch
    data['training']['epochs'] = epochs
    if max_steps is not None:
        data['training']['max_train_steps'] = max_steps
    else:
        data['training'].pop('max_train_steps', None)
    tmp.write_text(yaml.safe_dump(data, allow_unicode=True))

def run(cfg_path: Path) -> int:
    proc = subprocess.run([sys.executable,'scripts/run_experiment.py','--config',str(cfg_path),'--fold','0'])
    return proc.returncode

def main():
    assert BASE_CFG.exists(), f"Missing {BASE_CFG}"
    bs_list = [6,5,4]
    # Dry-run 1 step preflight with 20GB threshold
    for bs in bs_list:
        tmp = Path('configs/_tmp_b7_pref.yaml')
        patch_cfg(tmp, bs, epochs=1, max_steps=1)
        before = nvsmi_mem_used()
        rc = run(tmp)
        after = nvsmi_mem_used()
        delta = max(0, after - before)
        print(json.dumps({'preflight_bs':bs,'mem_before_mib':before,'mem_after_mib':after,'delta_mib':delta}))
        tmp.unlink(missing_ok=True)
        if rc==0 and (before+delta) < 20000:  # <20GB
            chosen_bs = bs
            break
    else:
        print('Preflight failed or memory too high; lowering resolution is required.')
        sys.exit(2)

    # Full 6-epoch run
    full_cfg = Path('configs/_tmp_b7_full.yaml')
    patch_cfg(full_cfg, chosen_bs, epochs=6, max_steps=None)
    rc = run(full_cfg)
    full_cfg.unlink(missing_ok=True)
    if rc!=0:
        print('Full run exited with non-zero status')
        sys.exit(rc)

    # Find latest run dir
    runs = sorted((Path('outputs')/ 'runs').glob('*efficientnet_b7_offline_448*'))
    if not runs:
        print('No run dir found')
        sys.exit(3)
    run_dir = runs[-1]
    print(json.dumps({'run_dir':str(run_dir)}))

    # Generate submission with SwinIR noise15 + TS + per-class PNG
    swinir_w = '/root/SwinIR/experiments/pretrained_models/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth'
    subprocess.run([sys.executable,'scripts/create_submission.py','--run-dir',str(run_dir),
                    '--denoise','swinir','--swinir-weights',swinir_w,
                    '--suffix','effb7_448_step1_swinir_noise15','--skip-if-unchanged'], check=True)
    ts_json = f"reports/run_history/{int(time.time())}_effb7_448_fold0_ts.json"
    subprocess.run([sys.executable,'scripts/apply_temperature_scaling.py','--run-dir',str(run_dir),
                    '--temperature-json',ts_json,'--suffix','effb7_448_tscaled'], check=True)
    subprocess.run([sys.executable,'scripts/plot_per_class_correct_incorrect.py','--runs-root','outputs/runs','--dest','reports/per_class'], check=True)

if __name__=='__main__':
    main()

