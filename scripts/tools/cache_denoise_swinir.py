#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import sys
import math
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

# Reuse the denoiser implementation from create_submission for consistency
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from scripts.create_submission import SwinIRDenoiser  # type: ignore


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(arr: np.ndarray, path: Path, fmt: str = 'png', quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr)
    if fmt.lower() == 'png':
        im.save(path.with_suffix('.png'))
    else:
        im.save(path.with_suffix('.jpg'), quality=quality)


def tile_iter(h, w, tile, overlap):
    stride = tile - overlap
    ny = math.ceil((h - overlap) / stride)
    nx = math.ceil((w - overlap) / stride)
    for iy in range(ny):
        for ix in range(nx):
            y0 = max(0, iy * stride)
            x0 = max(0, ix * stride)
            y1 = min(h, y0 + tile)
            x1 = min(w, x0 + tile)
            yield y0, y1, x0, x1


def run_swinir(denoiser: SwinIRDenoiser, device, img: np.ndarray, tile: int, overlap: int, fp16: bool) -> np.ndarray:
    h, w, _ = img.shape
    out = np.zeros_like(img, dtype=np.uint8)
    acc = np.zeros((h, w, 1), dtype=np.float32)

    denoiser.model.eval()
    autocast = torch.cuda.amp.autocast if fp16 else torch.cuda.amp.autocast
    with torch.no_grad():
        for y0, y1, x0, x1 in tile_iter(h, w, tile, overlap):
            crop = img[y0:y1, x0:x1]
            tens = TF.to_tensor(Image.fromarray(crop)).unsqueeze(0).to(device)
            with autocast():
                # denoiser expects normalized tensors; mean=0, std=1 keeps [0,1] range
                mean = torch.zeros(1, 3, 1, 1, device=device)
                std = torch.ones(1, 3, 1, 1, device=device)
                pred = denoiser(tens, mean, std).cpu()
            pred = torch.clamp(pred, 0, 1)
            out_crop = (pred[0].mul(255).byte().permute(1, 2, 0).numpy())
            out[y0:y1, x0:x1] = out_crop
            acc[y0:y1, x0:x1, 0] += 1.0

    acc[acc == 0] = 1.0
    out = (out.astype(np.float32) / acc).astype(np.uint8)
    return out


def build_swinir(weights: str, device: torch.device):
    return SwinIRDenoiser(weights, device)


def main():
    ap = argparse.ArgumentParser(description='Build SwinIR dn15 cache for test images')
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--format', choices=['png', 'jpg'], default='png')
    ap.add_argument('--quality', type=int, default=95)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoiser = build_swinir(args.weights, device)

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = [p for p in sorted(test_dir.iterdir()) if p.suffix.lower() in exts]
    print(f"Found {len(files)} images in {test_dir}")

    for idx, src in enumerate(files, 1):
        base = src.stem
        dst = (out_dir / base).with_suffix('.png' if args.format=='png' else '.jpg')
        if dst.exists() and not getattr(args, 'overwrite', False):
            continue
        img = load_image(src)
        den = run_swinir(denoiser, device, img, args.tile, args.overlap, args.fp16)
        save_image(den, dst, fmt=args.format, quality=args.quality)
        if idx % 25 == 0:
            print(f"Cached {idx}/{len(files)}: {dst}")

    print("Done. Cache saved to:", out_dir)


if __name__ == '__main__':
    main()
