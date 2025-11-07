#!/usr/bin/env python
"""
Offline augmentation script (Augraphy + geometric ops).

생성 순서
---------
1. train.csv / 이미지를 읽고 클래스별 프로필(card_like / non_doc / doc)을 결정한다.
2. 각 프로필에 대해 AugraphyPipeline을 구성한다.
   - Markup(취소선/밑줄/하이라이트)
   - Scribbles(낙서)
   - ColorPaper(용지 색상)
   - InkBleed + BleedThrough(잉크/블리드)
   - Folding(접기)
   - LightingGradient / ShadowCast(조명+그림자)
   - Geometric + Rescale(크기/회전/뒤집기)
   - Jpeg(압축)
3. target_total에 맞춰 클래스별 증강 수를 균등 분배하고, 이미지별로 반복 횟수를 계획한다.
4. 증강 이미지를 `output_dir/images/`에 저장하고, 병합 CSV(`train_offline_aug.csv`)와 요약표를 생성한다.

사용법
------
python scripts/offline_augment.py --config configs/offline_aug.yaml \\
    --train-csv data/raw/train.csv --image-dir data/raw/train

메모리 주의
-----------
한 번에 하나의 이미지만 처리하고, 결과 배열을 즉시 디스크에 쓰도록 구현하여 GPU/CPU 메모리 사용량을 최소화했다.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from augraphy import (
    AugraphyPipeline,
    BleedThrough,
    ColorPaper,
    Folding,
    Geometric,
    InkBleed,
    Jpeg,
    LightingGradient,
    Markup,
    NoiseTexturize,
    Rescale,
    Scribbles,
    ShadowCast,
)
from tqdm.auto import tqdm


def load_cfg(path: Path) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_profile(target: int, profiles: Dict[str, List[int]]) -> str:
    for name, ids in profiles.items():
        if target in ids:
            return name
    return "doc"


def plan_repeats_per_image(
    df: pd.DataFrame, target_total: int, repeats_min_per_image: int
) -> Dict[str, int]:
    """클래스 균등 분배(target_total/n_classes) 기준으로 이미지별 증강 횟수 결정."""
    n_classes = df["target"].nunique()
    per_class_goal = math.ceil(target_total / n_classes)
    counts = df["target"].value_counts().to_dict()

    per_image_repeats: Dict[str, int] = {}
    by_class: Dict[int, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        by_class[int(row["target"])].append(row["ID"])

    for cls, ids in by_class.items():
        need = max(0, per_class_goal - counts.get(cls, 0))
        if not ids:
            continue
        base, rem = divmod(need, len(ids))
        for idx, img in enumerate(ids):
            extra = 1 if idx < rem else 0
            per_image_repeats[img] = max(repeats_min_per_image, base + extra)
    return per_image_repeats


def random_markup(profile: str) -> Markup:
    return Markup(
        num_lines_range=(2, 6) if profile != "card_like" else (1, 3),
        markup_length_range=(0.4, 1.0),
        markup_thickness_range=(1, 3) if profile != "card_like" else (1, 2),
        markup_type="random",
        markup_ink="random",
        markup_color="random",
        repetitions=1,
        p=0.5 if profile != "non_doc" else 0.15,
    )


def random_scribbles(profile: str) -> Scribbles:
    return Scribbles(
        scribbles_type="lines",
        scribbles_size_range=(320, 700),
        scribbles_count_range=(2, 6) if profile != "card_like" else (1, 3),
        scribbles_thickness_range=(1, 3),
        scribbles_brightness_change=[48, 64, 96],
        scribbles_skeletonize=0,
        scribbles_text=None,
        scribbles_text_font=None,
        p=0.3 if profile != "non_doc" else 0.1,
    )


def build_pipeline(profile: str, rng: np.random.Generator) -> AugraphyPipeline:
    """프로필별 AugraphyPipeline."""
    ink_phase = []
    if profile != "non_doc":
        ink_phase.extend(
            [
                random_markup(profile),
                random_scribbles(profile),
                InkBleed(
                    intensity_range=(0.25, 0.5) if profile == "doc" else (0.2, 0.35),
                    kernel_size=(5, 5),
                    severity=(0.25, 0.4),
                    p=0.35,
                ),
            ]
        )

    paper_phase: List = []
    if profile != "non_doc":
        paper_phase.append(
            ColorPaper(
                hue_range=(20, 40) if profile == "doc" else (15, 30),
                saturation_range=(8, 25),
                p=0.7,
            )
        )
        paper_phase.append(NoiseTexturize(sigma_range=(2, 6), turbulence_range=(2, 4), p=0.4))

    post_phase = []
    if profile != "non_doc":
        post_phase.append(
            BleedThrough(
                intensity_range=(0.05, 0.3),
                color_range=(0, 200),
                alpha=0.25,
                offsets=(16, 16),
                p=0.2,
            )
        )
        post_phase.append(
            Folding(
                fold_count=1 if profile == "card_like" else 2,
                fold_angle_range=(-6, 6),
                gradient_width=(0.1, 0.2),
                gradient_height=(0.01, 0.03),
                p=0.15,
            )
        )

    post_phase.extend(
        [
            LightingGradient(
                mode="gaussian",
                max_brightness=255,
                min_brightness=120,
                transparency=None,
                p=0.35,
            ),
            ShadowCast(
                shadow_opacity_range=(0.2, 0.45),
                shadow_blur_kernel_range=(51, 151),
                p=0.2,
            ),
            Geometric(
                scale=(0.95, 1.05),
                translation=(0.02, 0.05),
                fliplr=0.3 if profile != "non_doc" else 0.2,
                flipud=0.0,
                rotate_range=(-12, 12),
                padding=[0, 0, 0, 0],
                randomize=1,
                p=1.0,
            ),
            Rescale(target_dpi=int(rng.integers(260, 330))),
            Jpeg(quality_range=(80, 96), p=0.25 if profile != "card_like" else 0.2),
        ]
    )

    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        overlay_alpha=0.3,
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )
    return pipeline


def augment_and_save(
    img_path: Path,
    profile: str,
    repeats: int,
    save_dir: Path,
    tag: str,
    rng: np.random.Generator,
) -> List[str]:
    data = np.fromfile(str(img_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return []
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    saved: List[str] = []
    for idx in range(repeats):
        pipeline = build_pipeline(profile, rng)
        result = pipeline.augment(image_rgb.copy())
        augmented = result.get("output")
        if augmented is None:
            continue
        out_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        name = f"{img_path.stem}__aug{idx}_{tag}.jpg"
        out_path = save_dir / name
        success, buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            continue
        out_path.write_bytes(buf.tobytes())
        saved.append(name)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/offline_aug.yaml")
    parser.add_argument("--train-csv", type=str, default="data/raw/train.csv")
    parser.add_argument("--image-dir", type=str, default="data/raw/train")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="처리할 이미지 개수 (없으면 전체)")
    parser.add_argument("--offset", type=int, default=0, help="앞쪽에서 건너뛸 이미지 수")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "data/processed/offline_aug"))
    tag = args.tag or cfg.get("save_tag", "offline")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    image_dir = Path(args.image_dir)

    repeats_map = plan_repeats_per_image(
        df=df,
        target_total=int(cfg.get("target_total", 25000)),
        repeats_min_per_image=int(cfg.get("repeats_min_per_image", 2)),
    )

    profiles = cfg.get("profiles", {})
    rng = np.random.default_rng(int(cfg.get("seed", 2024)))

    records = []
    if args.limit is None:
        selected = df.iloc[args.offset :]
    else:
        start = max(0, args.offset)
        end = min(len(df), start + args.limit)
        selected = df.iloc[start:end]

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="offline-aug"):
        img_id = row["ID"]
        target = int(row["target"])
        repeats = int(repeats_map.get(img_id, cfg.get("repeats_min_per_image", 2)))
        profile = select_profile(target, profiles)
        src = image_dir / img_id
        saved = augment_and_save(src, profile, repeats, images_dir, tag, rng)
        for name in saved:
            records.append({"ID": name, "target": target, "source": img_id, "profile": profile})

    aug_df = pd.DataFrame.from_records(records)
    merged = pd.concat([df.copy(), aug_df], ignore_index=True)
    merged_csv = output_dir / "train_offline_aug.csv"
    merged.to_csv(merged_csv, index=False)

    summary = aug_df.groupby(["target", "profile"]).size().reset_index(name="aug_count")
    summary_path = output_dir / "offline_aug_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Augmented images:", len(aug_df))
    print("Merged CSV:", merged_csv)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
