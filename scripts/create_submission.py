#!/usr/bin/env python

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.dataset import DocumentDataset
from src.models.factory import create_model
from src.transforms.factory import create_transforms
from src.utils.run_naming import build_submission_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission CSV.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config (optional if --run-dir provided).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Training run directory containing config/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Specific checkpoint path. Defaults to run_dir/checkpoints/best.pth",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        default="data/raw/sample_submission.csv",
        help="Path to sample submission file.",
    )
    parser.add_argument(
        "--leaderboard-score",
        type=float,
        default=None,
        help="Leaderboard score to embed in filename.",
    )
    parser.add_argument(
        "--save-probs",
        action="store_true",
        help="Save softmax probabilities alongside CSV.",
    )
    parser.add_argument(
        "--denoise",
        type=str,
        default=None,
        choices=[None, "swinir", "swinir_proxy"],
        help="Optional test-time denoising backend.",
    )
    parser.add_argument(
        "--swinir-weights",
        type=str,
        default=None,
        help="Path to SwinIR weights (.pth) when using --denoise swinir.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix to append to submission filename (e.g., tta90_hflip).",
    )
    parser.add_argument(
        "--skip-if-unchanged",
        action="store_true",
        help="If a previous submission for the same run has identical predictions, skip writing a new CSV.",
    )
    parser.add_argument(
        "--denoise-cache-dir",
        type=str,
        default=None,
        help="Optional directory containing precomputed denoised test images (PNG/JPG). If provided, loader will prefer cache over raw.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.run_dir:
        config_path = Path(args.run_dir) / "config.yaml"
    elif args.config:
        config_path = Path(args.config)
    else:
        raise ValueError("Either --run-dir or --config must be provided.")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> float:
    state = torch.load(checkpoint_path, map_location=device)
    metric = state.get("metric", 0.0)
    ema_state = state.get("ema_state")
    if ema_state:
        model.load_state_dict(ema_state, strict=False)
    else:
        swa_state = state.get("swa_state")
        if swa_state:
            model.load_state_dict(swa_state, strict=False)
        else:
            model_state = state.get("model_state") or state
            model.load_state_dict(model_state, strict=False)
    return metric


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    run_dir = Path(args.run_dir) if args.run_dir else None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is None:
        if run_dir is None:
            raise ValueError("Provide --checkpoint when --run-dir is absent.")
        checkpoint_path = run_dir / "checkpoints" / "best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(cfg)
    metric_value = load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    test_transforms = create_transforms(cfg, is_train=False)

    sample_df = pd.read_csv(args.sample_submission)
    test_dataset = DocumentDataset(
        dataframe=sample_df[["ID"]].copy(),
        image_dir=Path(cfg["paths"]["test_dir"]),
        transforms=test_transforms,
        is_train=False,
        cache_dir=Path(args.denoise_cache_dir) if args.denoise_cache_dir else None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["data"]["loader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["loader"]["num_workers"],
        pin_memory=cfg["data"]["loader"].get("pin_memory", False),
        persistent_workers=cfg["data"]["loader"].get("persistent_workers", False),
        prefetch_factor=(
            cfg["data"]["loader"].get("prefetch_factor", 2)
            if cfg["data"]["loader"].get("num_workers", 0) > 0
            else None
        ),
    )

    mean = torch.tensor(cfg["data"].get("mean", [0.0, 0.0, 0.0]), dtype=torch.float32).view(1, -1, 1, 1).to(device)
    std = torch.tensor(cfg["data"].get("std", [1.0, 1.0, 1.0]), dtype=torch.float32).view(1, -1, 1, 1).to(device)

    swinir_denoiser = None
    if args.denoise == "swinir":
        weights_path = args.swinir_weights or os.environ.get("SWINIR_WEIGHTS")
        if not weights_path:
            raise ValueError("--swinir-weights or SWINIR_WEIGHTS env var must be provided when using SwinIR denoising")
        swinir_denoiser = SwinIRDenoiser(weights_path, device)

    all_probs = []
    image_ids = []
    tta_cfg = cfg.get("submission", {}).get("tta", {})
    tta_enabled = tta_cfg.get("enabled", False)
    rotations = tta_cfg.get("rotations", [0]) if tta_enabled else [0]
    hflip_enabled = bool(tta_cfg.get("horizontal_flip", False))
    vflip_enabled = bool(tta_cfg.get("vertical_flip", False))
    jitter = float(tta_cfg.get("jitter_degrees", 0) or 0)

    tta_angles = []
    for base in rotations:
        base = float(base)
        tta_angles.append(base)
        if jitter > 0:
            tta_angles.extend([base - jitter, base + jitter])
    if not tta_angles:
        tta_angles = [0.0]
    # Normalize angles to range [-180, 180) to avoid redundant rotations
    normalized_angles = []
    for angle in tta_angles:
        a = ((angle + 180) % 360) - 180
        normalized_angles.append(round(a, 2))
    unique_angles = sorted(set(normalized_angles))

    flip_options = [(False, False)]
    if hflip_enabled or vflip_enabled:
        flip_options = []
        h_options = [False, True] if hflip_enabled else [False]
        v_options = [False, True] if vflip_enabled else [False]
        for hf in h_options:
            for vf in v_options:
                if hf or vf:
                    flip_options.append((hf, vf))
        if (False, False) not in flip_options:
            flip_options.insert(0, (False, False))

    use_denoise = args.denoise == "swinir_proxy"
    use_swinir = args.denoise == "swinir"
    if use_denoise:
        import cv2

        def _denoise_batch(t: torch.Tensor) -> torch.Tensor:
            restored = (t * std + mean).clamp(0, 1)
            b, c, h, w = restored.shape
            out = []
            t_cpu = restored.detach().cpu()
            for i in range(b):
                img = t_cpu[i].permute(1, 2, 0).numpy()
                img8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
                den = cv2.bilateralFilter(img8, d=5, sigmaColor=25, sigmaSpace=7)
                denf = den.astype(np.float32) / 255.0
                out.append(torch.from_numpy(denf).permute(2, 0, 1))
            out_t = torch.stack(out, dim=0).to(t.device)
            return (out_t - mean) / std

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, non_blocking=True)
            if use_denoise:
                images = _denoise_batch(images)
            if use_swinir and swinir_denoiser is not None:
                images = swinir_denoiser(images, mean, std)
            tta_probs = []
            for angle in unique_angles:
                if abs(angle) < 1e-4:
                    rotated = images
                else:
                    mod_angle = angle % 360
                    # Use fast integer rotations when possible
                    if abs((mod_angle % 90)) < 1e-4:
                        k = int(round(mod_angle / 90)) % 4
                        rotated = torch.rot90(images, k=k, dims=(2, 3))
                    else:
                        rotated = torch.stack(
                            [
                                F.rotate(
                                    img,
                                    angle,
                                    interpolation=InterpolationMode.BILINEAR,
                                    fill=0.0,
                                )
                                for img in images
                            ],
                            dim=0,
                        )
                for hf, vf in flip_options:
                    aug = rotated
                    if hf:
                        aug = torch.flip(aug, dims=(3,))
                    if vf:
                        aug = torch.flip(aug, dims=(2,))
                    outputs = model(aug)
                    tta_probs.append(torch.softmax(outputs, dim=1))
            if len(tta_probs) == 1:
                probs = tta_probs[0]
            else:
                probs = torch.stack(tta_probs, dim=0).mean(dim=0)
            all_probs.append(probs.cpu().numpy())
            image_ids.extend(ids)

    probs_array = np.vstack(all_probs)
    preds = probs_array.argmax(axis=1)

    assert list(sample_df["ID"]) == image_ids, "Prediction order does not match sample submission."

    submission_df = sample_df.copy()
    submission_df["target"] = preds

    leaderboard_score = (
        args.leaderboard_score
        if args.leaderboard_score is not None
        else cfg["submission"].get("leaderboard_score", 0.0)
    )
    base_name = run_dir.name if run_dir is not None else None
    submission_name = build_submission_name(
        cfg,
        metric_value,
        leaderboard_score,
        base_name=base_name,
    )
    if args.suffix:
        submission_name = f"{submission_name}_{args.suffix}"

    output_dir = Path(cfg["paths"]["output_root"]) / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{submission_name}.csv"
    # Optional: skip writing if predictions unchanged vs latest CSV of the same run
    if args.skip_if_unchanged and run_dir is not None:
        import hashlib
        output_dir = Path(cfg["paths"]["output_root"]) / "submissions"
        def _hash_targets(df: pd.DataFrame) -> str:
            # Hash in the order of sample submission IDs
            data = (df["ID"].astype(str) + "," + df["target"].astype(str)).str.cat(sep="\n").encode()
            return hashlib.sha256(data).hexdigest()

        this_sig = _hash_targets(submission_df)
        # find latest CSV with same run base name (prefix)
        candidates = sorted(
            [p for p in output_dir.glob(f"{run_dir.name}_*.csv")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for prev in candidates:
            try:
                prev_df = pd.read_csv(prev)
                prev_sig = _hash_targets(prev_df)
                if prev_sig == this_sig:
                    print(f"Predictions unchanged vs {prev.name}. Skipping write due to --skip-if-unchanged.")
                    return
            except Exception:
                continue

    submission_df.to_csv(csv_path, index=False)

    if args.save_probs:
        probs_dir = Path(cfg["paths"]["output_root"]) / "logits"
        probs_dir.mkdir(parents=True, exist_ok=True)
        np.save(probs_dir / f"{submission_name}.npy", probs_array)

    print(f"Saved submission to {csv_path}")


class SwinIRDenoiser:
    """Thin wrapper to run SwinIR denoising inside submission generation."""

    def __init__(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        self.weights_path = Path(weights_path)
        swinir_root = Path("/root/SwinIR")
        if swinir_root.exists() and str(swinir_root) not in sys.path:
            sys.path.append(str(swinir_root))
        from models.network_swinir import SwinIR as SwinIRModel  # type: ignore

        config = self._infer_config(self.weights_path.name.lower())
        self.config = config
        self.model = self._build_model(SwinIRModel, config).to(device)
        state = torch.load(self.weights_path, map_location=device)
        key = config.get("param_key", "params")
        if key in state:
            state = state[key]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @staticmethod
    def _infer_config(name: str) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "scale": 1,
            "img_range": 1.0,
            "window_size": 8,
            "in_chans": 3,
            "task": "color_dn",
            "param_key": "params",
        }
        noise_match = re.search(r"noise(\\d+)", name)
        jpeg_match = re.search(r"jpeg(\\d+)", name)
        if "colordn" in name:
            cfg["task"] = "color_dn"
            cfg["noise"] = int(noise_match.group(1)) if noise_match else 25
        elif "colorcar" in name:
            cfg["task"] = "color_jpeg_car"
            cfg["jpeg"] = int(jpeg_match.group(1)) if jpeg_match else 40
            cfg["img_range"] = 255.0
            cfg["window_size"] = 7
        elif "jpeg" in name:
            cfg["task"] = "jpeg_car"
            cfg["jpeg"] = int(jpeg_match.group(1)) if jpeg_match else 40
            cfg["img_range"] = 255.0
            cfg["window_size"] = 7
            cfg["in_chans"] = 1
        else:
            raise ValueError(f"Unsupported SwinIR weights naming: {name}")
        return cfg

    def _build_model(self, model_cls, cfg: Dict[str, Any]):
        common_kwargs = dict(
            upscale=cfg["scale"],
            in_chans=cfg["in_chans"],
            img_size=128 if cfg["window_size"] == 8 else 126,
            window_size=cfg["window_size"],
            img_range=cfg["img_range"],
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv',
        )
        model = model_cls(**common_kwargs)
        return model

    def __call__(self, images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """images: normalized tensor (B,C,H,W)."""
        orig = (images * std + mean).clamp(0, 1)
        denoised = self._run_inference(orig)
        denoised = denoised.clamp(0, 1)
        return (denoised - mean) / std

    def _run_inference(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        cfg = self.config
        original_channels = x.shape[1]
        if cfg["in_chans"] == 1 and original_channels == 3:
            # convert to luminance
            luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            x = luma
        if cfg["img_range"] == 255.0:
            x = x * 255.0

        _, _, h0, w0 = x.shape
        pad_h = (cfg["window_size"] - h0 % cfg["window_size"]) % cfg["window_size"]
        pad_w = (cfg["window_size"] - w0 % cfg["window_size"]) % cfg["window_size"]
        if pad_h > 0:
            x = torch.cat([x, torch.flip(x, [2])], dim=2)[:, :, : h0 + pad_h, :]
        if pad_w > 0:
            x = torch.cat([x, torch.flip(x, [3])], dim=3)[:, :, :, : w0 + pad_w]

        with torch.no_grad():
            out = self.model(x)

        out = out[..., :h0, :w0]
        if cfg["img_range"] == 255.0:
            out = out / 255.0
        if cfg["in_chans"] == 1 and original_channels == 3:
            out = out.repeat(1, 3, 1, 1)
        return out


if __name__ == "__main__":
    main()
