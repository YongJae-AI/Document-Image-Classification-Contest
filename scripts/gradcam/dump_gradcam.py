#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.factory import create_model
from src.transforms.factory import create_transforms
from src.data.dataset import DocumentDataset


def parse_args():
    ap = argparse.ArgumentParser(description="Dump Grad-CAM for misclassified validation samples (per-class Top-N)")
    ap.add_argument('--run-dir', required=True, help='Training run directory containing config/checkpoints')
    ap.add_argument('--topk', type=int, default=10, help='Max samples per class to dump')
    ap.add_argument('--out-root', type=str, default='reports/gradcam', help='Output root directory')
    ap.add_argument('--mode', type=str, default='incorrect', choices=['incorrect','correct'], help='Which samples to visualize per class')
    ap.add_argument('--focus-classes', type=str, default=None, help='Comma-separated class ids to restrict dumping (e.g., "3,14")')
    ap.add_argument('--target', type=str, default='gt', choices=['gt','pred'], help='Grad-CAM target: ground-truth or predicted class')
    ap.add_argument('--group-by', type=str, default='gt', choices=['gt','pred'], help='Group samples per class by ground-truth or predicted class when selecting')
    ap.add_argument('--dpi', type=int, default=150)
    return ap.parse_args()


def load_cfg(run_dir: Path) -> Dict[str, Any]:
    with open(run_dir / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module | None:
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._save_activations)
        self._h2 = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._h1.remove(); self._h2.remove()

    @torch.no_grad()
    def _to_heatmap(self, cam: torch.Tensor) -> np.ndarray:
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam_np = (cam.cpu().numpy() * 255.0).astype(np.uint8)
        return cam_np

    def __call__(self, images: torch.Tensor, targets: torch.Tensor | None = None) -> List[np.ndarray]:
        # Forward
        logits = self.model(images)
        if hasattr(logits, 'logits'):
            logits = logits.logits
        if targets is None:
            targets = logits.argmax(dim=1)
        # Backward for each sample
        cams: List[np.ndarray] = []
        for i in range(images.size(0)):
            self.model.zero_grad(set_to_none=True)
            score = logits[i, targets[i]]
            score.backward(retain_graph=True)
            # GradCAM: weights = GAP over gradients
            grads = self.gradients[i]  # (C, H, W)
            acts = self.activations[i]
            weights = grads.mean(dim=(1, 2))  # (C,)
            cam = torch.zeros_like(acts[0])
            for w, a in zip(weights, acts):
                cam += w * a
            cam = torch.relu(cam)
            cam_np = self._to_heatmap(cam)
            cams.append(cam_np)
        return cams


def overlay_cam_on_image(img_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    h, w, _ = img_rgb.shape
    cam_resized = cv2.resize(cam, (w, h))
    heat = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    over = (0.4 * heat + 0.6 * img_rgb).astype(np.uint8)
    return over


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_cfg(run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and load best checkpoint
    from src.models.factory import create_model
    model = create_model(cfg).to(device)
    state = torch.load(run_dir / 'checkpoints' / 'best.pth', map_location=device)
    model_state = state.get('ema_state') or state.get('swa_state') or state.get('model_state') or state
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # Target layer for Grad-CAM
    target_layer = find_last_conv_module(model)
    if target_layer is None:
        print('No Conv2d layer found — skipping Grad-CAM for this run (likely transformer-only).')
        return
    cammer = GradCAM(model, target_layer, device)

    # Build validation dataset
    import pickle
    split_pkl = Path(cfg['split']['predefined_split'])
    with open(split_pkl, 'rb') as f:
        folds = pickle.load(f)
    fold_idx = int(cfg['split']['fold_index'])
    if isinstance(folds, list):
        train_idx, val_idx = folds[fold_idx]
    else:
        raise ValueError('Unsupported split format')
    import pandas as pd
    df = pd.read_csv(cfg['paths']['train_csv'])
    val_df = df.iloc[val_idx].reset_index(drop=True)
    tfm = create_transforms(cfg, is_train=False)
    ds = DocumentDataset(val_df, Path(cfg['paths']['image_dir']), transforms=tfm, is_train=True)

    # Run inference to collect indices per class by mode
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['data']['loader'].get('batch_size', 8), shuffle=False,
                                         num_workers=cfg['data']['loader'].get('num_workers', 4), pin_memory=True)
    num_classes = int(cfg['data']['num_classes'])
    per_cls_mis: List[List[int]] = [[] for _ in range(num_classes)]  # group by GT
    per_cls_cor: List[List[int]] = [[] for _ in range(num_classes)]  # group by GT
    per_pred_mis: List[List[int]] = [[] for _ in range(num_classes)]  # group by predicted class
    per_pred_cor: List[List[int]] = [[] for _ in range(num_classes)]  # group by predicted class
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            out = model(x)
            if hasattr(out, 'logits'):
                out = out.logits
            pred = out.argmax(1).cpu()
            y_cpu = y.cpu()
            for i in range(len(y_cpu)):
                yi = int(y_cpu[i])
                pi = int(pred[i])
                if pi != yi:
                    per_cls_mis[yi].append(idx * loader.batch_size + i)
                    per_pred_mis[pi].append(idx * loader.batch_size + i)
                else:
                    per_cls_cor[yi].append(idx * loader.batch_size + i)
                    per_pred_cor[pi].append(idx * loader.batch_size + i)

    # Output dir with unique stamp and run name
    stamp = state.get('epoch', None)
    out_dir = Path(args.out_root) / f"{run_dir.name}_gradcam_epoch{stamp if stamp is not None else 'best'}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine focus classes
    if args.focus_classes:
        try:
            focus = [int(t.strip()) for t in args.focus_classes.split(',') if t.strip()!='']
        except Exception:
            focus = []
    else:
        focus = list(range(num_classes))

    # Dump per-class Top-K
    for cls_id in focus:
        if args.group_by == 'pred':
            pool_cor = per_pred_cor
            pool_mis = per_pred_mis
        else:
            pool_cor = per_cls_cor
            pool_mis = per_cls_mis
        if args.mode == 'incorrect':
            ids_list = pool_mis[cls_id][: args.topk]
        else:
            ids_list = pool_cor[cls_id][: args.topk]
        if not ids_list:
            continue
        cls_dir = out_dir / f"class_{cls_id:02d}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for rank, ds_index in enumerate(ids_list, start=1):
            # Load raw image for overlay
            row = val_df.iloc[ds_index]
            img_path = Path(cfg['paths']['image_dir']) / row['ID']
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Build tensor via transforms
            sample = DocumentDataset(val_df.iloc[[ds_index]].reset_index(drop=True), Path(cfg['paths']['image_dir']), transforms=tfm, is_train=True)
            x_t, y_t = next(iter(torch.utils.data.DataLoader(sample, batch_size=1)))
            x_t = x_t.to(device)
            if args.target == 'pred':
                with torch.no_grad():
                    logits = model(x_t)
                    if hasattr(logits, 'logits'):
                        logits = logits.logits
                    targ = logits.argmax(1).to(device)
                cams = cammer(x_t, targets=targ)
            else:
                cams = cammer(x_t, targets=y_t.to(device))
            over = overlay_cam_on_image(rgb, cams[0])
            # Unique, informative filename
            fname = f"{run_dir.name}_{args.mode}_cls{cls_id:02d}_rank{rank:02d}_{row['ID'].replace('.jpg','')}.png"
            cv2.imwrite(str(cls_dir / fname), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

    cammer.remove()
    print('Grad-CAM saved to', out_dir)


if __name__ == '__main__':
    main()
