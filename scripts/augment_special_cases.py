#!/usr/bin/env python

"""
Generate additional training samples for special-case document images.

This script reads the metadata of excluded images, applies deterministic
augmentation pipelines (CLAHE, MotionBlur, ISO Noise, Perspective, etc.),
and saves the augmented copies back into the main training directory.

Each augmented file is registered in train.csv with the same target label.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2
import pandas as pd


SPECIAL_IMAGE_IDS = [
    "be53872196b3ae1d.jpg",
    "f0fc4e2f239e236b.jpg",
    "f176f6c25b7bd8ae.jpg",
]


def build_transforms() -> Dict[str, A.Compose]:
    return {
        "clahe": A.Compose(
            [
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            ]
        ),
        "motion_blur": A.Compose(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ]
        ),
        "iso_noise": A.Compose(
            [
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ]
        ),
        "perspective": A.Compose(
            [
                A.Perspective(scale=(0.05, 0.1), keep_size=True, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment special-case training images.")
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/raw/train",
        help="Directory where augmented files will be stored.",
    )
    parser.add_argument(
        "--excluded-dir",
        type=str,
        default="data/raw/excluded_images",
        help="Directory containing the special-case original images.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/raw/train.csv",
        help="CSV file containing training metadata.",
    )
    parser.add_argument(
        "--excluded-csv",
        type=str,
        default="data/raw/excluded_samples.csv",
        help="CSV file tracking excluded samples and labels.",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="data/raw/special_augmented_samples.csv",
        help="CSV file to record augmented sample metadata.",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image) -> None:
    ensure_directory(path.parent)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_dir)
    excluded_dir = Path(args.excluded_dir)
    train_csv_path = Path(args.train_csv)
    excluded_csv_path = Path(args.excluded_csv)
    log_csv_path = Path(args.log_csv)

    ensure_directory(train_dir)

    if not train_csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")
    if not excluded_csv_path.exists():
        raise FileNotFoundError(f"excluded_samples.csv not found at {excluded_csv_path}")

    transforms = build_transforms()

    train_df = pd.read_csv(train_csv_path)
    excluded_df = pd.read_csv(excluded_csv_path)
    log_records: List[Dict[str, str]] = []

    for image_id in SPECIAL_IMAGE_IDS:
        row = excluded_df[excluded_df["ID"] == image_id]
        if row.empty:
            print(f"[!] Metadata for {image_id} not found in excluded_samples.csv; skipping.")
            continue

        target = int(row.iloc[0]["target"])
        source_path = excluded_dir / image_id
        if not source_path.exists():
            print(f"[!] Source image missing: {source_path}; skipping.")
            continue

        base_name = Path(image_id).stem

        # Copy original back into train directory if absent.
        target_original_path = train_dir / image_id
        if not target_original_path.exists():
            original_image = load_image(source_path)
            save_image(target_original_path, original_image)
            train_df = pd.concat(
                [train_df, pd.DataFrame([{"ID": image_id, "target": target}])],
                ignore_index=True,
            )
            log_records.append({"ID": image_id, "target": target, "variant": "original"})
            print(f"[+] Restored original image: {target_original_path}")

        original_image = load_image(source_path)

        for variant_name, transform in transforms.items():
            augmented = transform(image=original_image)["image"]
            variant_filename = f"{base_name}_aug_{variant_name}.jpg"
            variant_path = train_dir / variant_filename

            counter = 1
            while variant_path.exists():
                variant_filename = f"{base_name}_aug_{variant_name}_{counter}.jpg"
                variant_path = train_dir / variant_filename
                counter += 1

            save_image(variant_path, augmented)
            train_df = pd.concat(
                [train_df, pd.DataFrame([{"ID": variant_filename, "target": target}])],
                ignore_index=True,
            )
            log_records.append(
                {
                    "ID": variant_filename,
                    "target": target,
                    "variant": variant_name,
                    "source": image_id,
                }
            )
            print(f"[+] Generated {variant_name} variant for {image_id}")

    train_df = train_df.drop_duplicates(subset="ID", keep="last").reset_index(drop=True)
    train_df.to_csv(train_csv_path, index=False)

    if log_records:
        log_df = pd.DataFrame(log_records)
        if log_csv_path.exists():
            existing = pd.read_csv(log_csv_path)
            log_df = pd.concat([existing, log_df], ignore_index=True)
        log_df.to_csv(log_csv_path, index=False)
        print(f"[+] Logged {len(log_records)} entries to {log_csv_path}")
    else:
        print("[=] No new augmented samples were created.")


if __name__ == "__main__":
    main()
