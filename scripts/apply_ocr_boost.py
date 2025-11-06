#!/usr/bin/env python

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import easyocr
import numpy as np
import pandas as pd


TARGET_CLASSES = [5, 6, 8, 9, 10, 11, 12, 16]

# Simple keyword dictionary for OCR-based boosting.
OCR_KEYWORDS: Dict[int, Sequence[str]] = {
    5: ["DRIVER", "LICENCE", "LICENSE", "운전"],
    6: ["MEDICAL", "진료", "병원", "영수증"],
    8: ["NATIONAL ID", "ID CARD", "주민등록증", "IDENTIFICATION"],
    9: ["PASSPORT", "여권", "REPUBLIC OF"],
    10: ["PAYMENT", "납부", "CONFIRMATION", "완납"],
    11: ["PHARMACY", "약국", "PHARMACEUTICAL", "조제"],
    12: ["PRESCRIPTION", "처방"],
    16: ["VEHICLE", "REGISTRATION", "자동차등록증", "번호판"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply OCR-based probability boosts on ensemble outputs.")
    parser.add_argument(
        "--ensemble-csv",
        type=str,
        required=True,
        help="CSV produced by ensemble_softmax.py (ID ordering must follow sample_submission).",
    )
    parser.add_argument(
        "--ensemble-npy",
        type=str,
        required=True,
        help="Probability tensor (.npy) produced by ensemble_softmax.py.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="submissions",
        help="Directory where boosted CSV/NPY files are saved.",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=0.12,
        help="Additive boost applied to matching class probabilities before renormalisation.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["ko", "en"],
        help="Languages passed to easyocr.Reader.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="outputs/ocr_cache",
        help="Directory for storing cached OCR text outputs.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for easyocr (ensure GPU memory headroom before enabling).",
    )
    return parser.parse_args()


def needs_ocr(probs: np.ndarray) -> bool:
    top2 = probs.argsort()[::-1][:2]
    return any(cls in TARGET_CLASSES for cls in top2)


def main() -> None:
    args = parse_args()

    probs = np.load(args.ensemble_npy)
    df = pd.read_csv(args.ensemble_csv)
    sample_df = pd.read_csv("data/raw/sample_submission.csv")

    if not (len(df) == len(sample_df) == len(probs)):
        raise ValueError("Length mismatch among ensemble csv, sample submission, and probability tensor.")
    if not (df["ID"].tolist() == sample_df["ID"].tolist()):
        raise ValueError("ID ordering mismatch between ensemble csv and sample submission.")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(args.languages, gpu=args.gpu)

    boost_counts = defaultdict(int)
    image_dir = Path("data/raw/test")

    for idx, row in df.iterrows():
        prob_row = probs[idx]
        top2 = prob_row.argsort()[::-1][:2]
        candidate_classes = [cls for cls in top2 if cls in TARGET_CLASSES]
        if not candidate_classes:
            continue

        image_path = image_dir / row["ID"]
        if not image_path.exists():
            continue

        cache_path = cache_dir / f"{row['ID']}.txt"
        if cache_path.exists():
            raw_text = cache_path.read_text(encoding="utf-8").strip()
        else:
            ocr_texts: List[str] = reader.readtext(str(image_path), detail=0)
            raw_text = " ".join(ocr_texts)
            cache_path.write_text(raw_text, encoding="utf-8")

        joined_text = raw_text.upper()
        if not joined_text:
            continue

        applied = False
        for cls in candidate_classes:
            keywords = OCR_KEYWORDS.get(cls, [])
            if any(keyword.upper() in joined_text for keyword in keywords):
                probs[idx, cls] += args.boost
                boost_counts[cls] += 1
                applied = True

        if applied:
            probs[idx] = probs[idx] / probs[idx].sum()

    preds = probs.argmax(axis=1)
    boosted_df = sample_df.copy()
    boosted_df["target"] = preds

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(args.ensemble_csv).stem
    csv_path = output_dir / f"{base_name}_ocr.csv"
    npy_path = output_dir / f"{base_name}_ocr.npy"

    boosted_df.to_csv(csv_path, index=False)
    np.save(npy_path, probs)

    print(f"Saved OCR-boosted submission to {csv_path}")
    if boost_counts:
        for cls, count in sorted(boost_counts.items()):
            print(f"Class {cls}: applied boost on {count} samples")


if __name__ == "__main__":
    main()
