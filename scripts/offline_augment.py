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
    BrightnessTexturize,
    ColorPaper,
    Folding,
    Geometric,
    InkBleed,
    Jpeg,
    LightingGradient,
    LowLightNoise,
    Markup,
    Moire,
    NoiseTexturize,
    OneOf,
    PatternGenerator,
    ReflectedLight,
    Rescale,
    Scribbles,
    ShadowCast,
    VoronoiTessellation,
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
    df: pd.DataFrame,
    target_total: int,
    repeats_min_per_image: int,
    existing_counts: Dict[int, int] | None = None,
) -> Dict[str, int]:
    """클래스 균등 분배(target_total/n_classes) 기준으로 이미지별 증강 횟수 결정."""
    n_classes = df["target"].nunique()
    per_class_goal = math.ceil(target_total / n_classes)
    if existing_counts is None:
        counts = df["target"].value_counts().to_dict()
    else:
        counts = existing_counts

    per_image_repeats: Dict[str, int] = {}
    by_class: Dict[int, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        by_class[int(row["target"])].append(row["ID"])

    for cls, ids in by_class.items():
        need = max(0, per_class_goal - counts.get(cls, 0))
        if not ids or need == 0:
            for img in ids:
                per_image_repeats[img] = 0
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


def random_bleedthrough(profile: str, rng: np.random.Generator) -> BleedThrough | None:
    if profile == "non_doc":
        return None
    alpha = float(rng.uniform(0.08, 0.2))
    offsets = (
        int(rng.integers(6, 21)),
        int(rng.integers(6, 21)),
    )
    return BleedThrough(
        intensity_range=(0.08, 0.3) if profile == "doc" else (0.05, 0.25),
        color_range=(32, 224),
        ksize=(17, 17),
        sigmaX=1,
        alpha=alpha,
        offsets=offsets,
        p=0.35 if profile != "card_like" else 0.25,
    )


def random_color_paper(profile: str) -> ColorPaper:
    if profile == "non_doc":
        hue = (5, 12)
        saturation = (4, 10)
    elif profile == "card_like":
        hue = (10, 20)
        saturation = (6, 14)
    else:
        hue = (12, 28)
        saturation = (8, 20)
    return ColorPaper(
        hue_range=hue,
        saturation_range=saturation,
        p=0.3 if profile != "non_doc" else 0.4,
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
        paper_phase.append(random_color_paper(profile))
        paper_phase.append(
            OneOf(
                [
                    Moire(
                        moire_density=(15, 20),
                        moire_blend_method="normal",
                        moire_blend_alpha=0.1,
                        p=0.35,
                    ),
                    PatternGenerator(
                        imgx=int(rng.integers(256, 513)),
                        imgy=int(rng.integers(256, 513)),
                        n_rotation_range=(10, 15),
                        color="random",
                        alpha_range=(0.25, 0.5),
                        p=0.35,
                    ),
                    VoronoiTessellation(
                        mult_range=(50, 80),
                        num_cells_range=(500, 800),
                        noise_type="random",
                        background_value=(200, 256),
                        p=0.3,
                    ),
                ],
                p=0.3,
            )
        )
        paper_phase.append(
            OneOf(
                [
                    NoiseTexturize(
                        sigma_range=(5, 12),
                        turbulence_range=(3, 8),
                        texture_width_range=(80, 400),
                        texture_height_range=(80, 400),
                        p=0.5,
                    ),
                    BrightnessTexturize(texturize_range=(0.8, 0.99), deviation=0.02, p=0.5),
                ],
                p=0.65,
            )
        )
    else:
        paper_phase.append(random_color_paper(profile))

    post_phase: List = []
    bleed = random_bleedthrough(profile, rng)
    if bleed is not None:
        post_phase.append(bleed)
    if profile != "non_doc":
        post_phase.append(
            Folding(
                fold_count=1 if profile == "card_like" else 2,
                fold_angle_range=(-8, 8),
                gradient_width=(0.1, 0.2),
                gradient_height=(0.01, 0.03),
                p=0.2,
            )
        )

    post_phase.append(
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=90,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    transparency=0.5,
                    p=0.25,
                ),
                LowLightNoise(
                    num_photons_range=(60, 120),
                    alpha_range=(0.7, 0.9),
                    beta_range=(10, 30),
                    gamma_range=(1.0, 1.8),
                    p=0.25,
                ),
                ReflectedLight(
                    reflected_light_smoothness=0.8,
                    reflected_light_internal_radius_range=(0.0, 0.2),
                    reflected_light_external_radius_range=(0.1, 0.8),
                    reflected_light_minor_major_ratio_range=(0.9, 1.0),
                    reflected_light_color=(255, 255, 255),
                    reflected_light_internal_max_brightness_range=(0.9, 1.0),
                    reflected_light_external_max_brightness_range=(0.9, 0.95),
                    reflected_light_location="random",
                    reflected_light_ellipse_angle_range=(0, 360),
                    reflected_light_gaussian_kernel_size_range=(5, 151),
                    p=0.25,
                ),
                ShadowCast(
                    shadow_side="bottom",
                    shadow_vertices_range=(2, 3),
                    shadow_width_range=(0.5, 0.8),
                    shadow_height_range=(0.5, 0.8),
                    shadow_color=(0, 0, 0),
                    shadow_opacity_range=(0.4, 0.6),
                    shadow_iterations_range=(1, 2),
                    shadow_blur_kernel_range=(101, 201),
                    p=0.25,
                ),
            ],
            p=0.4,
        )
    )

    geom_options: List[Geometric] = [
        Geometric(
            scale=(0.92, 1.08),
            translation=(0.02, 0.06),
            fliplr=0.35 if profile != "non_doc" else 0.2,
            flipud=0.05,
            rotate_range=(-18, 18),
            padding=[0, 0, 0, 0],
            randomize=1,
            p=1.0,
        )
    ]
    if profile != "non_doc":
        geom_options.extend(
            [
                Geometric(
                    scale=(0.92, 1.05),
                    translation=(0.01, 0.03),
                    fliplr=0.1,
                    flipud=0.02,
                    rotate_range=(82, 98),
                    padding=[0, 0, 0, 0],
                    randomize=1,
                    p=1.0,
                ),
                Geometric(
                    scale=(0.92, 1.05),
                    translation=(0.01, 0.03),
                    fliplr=0.1,
                    flipud=0.02,
                    rotate_range=(-98, -82),
                    padding=[0, 0, 0, 0],
                    randomize=1,
                    p=1.0,
                ),
            ]
        )

    post_phase.append(OneOf(geom_options, p=1.0))
    post_phase.append(Rescale(target_dpi=int(rng.integers(260, 330))))
    post_phase.append(Jpeg(quality_range=(78, 95), p=0.35 if profile != "card_like" else 0.25))

    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        overlay_alpha=0.3,
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )
    return pipeline


def _existing_aug_count(save_dir: Path, img_stem: str, tag: str) -> int:
    existing = list(save_dir.glob(f"{img_stem}__aug*_{tag}.jpg"))
    max_idx = -1
    for path in existing:
        name = path.stem
        if "__aug" not in name:
            continue
        try:
            idx_part = name.split("__aug", 1)[1]
            idx = int(idx_part.split("_")[0])
            max_idx = max(max_idx, idx)
        except (ValueError, IndexError):
            continue
    return max_idx + 1


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
    next_idx = _existing_aug_count(save_dir, img_path.stem, tag)
    if repeats <= 0:
        return saved
    for idx in range(repeats):
        pipeline = build_pipeline(profile, rng)
        result = pipeline.augment(image_rgb.copy())
        augmented = result.get("output")
        if augmented is None:
            continue
        out_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        name = f"{img_path.stem}__aug{next_idx + idx}_{tag}.jpg"
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
    parser.add_argument("--existing-csv", type=str, default=None, help="이미 생성된 오프라인 CSV 경로")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "data/processed/offline_aug"))
    tag = args.tag or cfg.get("save_tag", "offline")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    image_dir = Path(args.image_dir)

    existing_counts = None
    existing_df = None
    existing_csv_path = args.existing_csv or cfg.get("existing_csv")
    if existing_csv_path and Path(existing_csv_path).exists():
        existing_df = pd.read_csv(existing_csv_path)
        existing_counts = existing_df["target"].value_counts().to_dict()

    repeats_map = plan_repeats_per_image(
        df=df,
        target_total=int(cfg.get("target_total", 25000)),
        repeats_min_per_image=int(cfg.get("repeats_min_per_image", 2)),
        existing_counts=existing_counts,
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
    if existing_df is not None:
        merged = pd.concat([existing_df, aug_df], ignore_index=True)
        merged.drop_duplicates(subset=["ID"], inplace=True, keep="last")
    else:
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
