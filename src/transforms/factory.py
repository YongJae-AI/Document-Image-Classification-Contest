import copy
from typing import Any, Dict, Iterable, List, Tuple

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from .divide_transforms import (
    DivideSixParts,
    DivideThreeParts,
    HalfDivide,
    QuarterDivide,
)


class TransformWrapper:
    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, *, image, label=None):
        return self.transform(image=image)


class TransformSelector(TransformWrapper):
    def __init__(
        self,
        default_transform: A.Compose,
        overrides: Iterable[Tuple[Iterable[int], A.Compose]],
    ) -> None:
        super().__init__(default_transform)
        self.overrides: List[Tuple[set[int], A.Compose]] = [
            (set(classes), transform) for classes, transform in overrides
        ]

    def __call__(self, *, image, label=None):
        transform = self.transform
        if label is not None:
            for classes, override in self.overrides:
                if label in classes:
                    transform = override
                    break
        return transform(image=image)


def create_transforms(cfg: Dict[str, Any], is_train: bool):
    size = cfg["data"]["input_size"]
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]

    if is_train:
        aug_cfg = copy.deepcopy(cfg["augmentations"])
        if aug_cfg.get("use_reference_pipeline"):
            return TransformWrapper(_build_reference_transform(aug_cfg, size, mean, std))
        class_overrides = aug_cfg.pop("class_overrides", [])

        default_transform = _build_train_transform(aug_cfg, size, mean, std)
        overrides: List[Tuple[Iterable[int], A.Compose]] = []

        for override in class_overrides:
            classes = override.get("classes", [])
            override_cfg = copy.deepcopy(aug_cfg)
            override_augments = override.get("augmentations", {})
            for key, value in override_augments.items():
                if value is None:
                    override_cfg.pop(key, None)
                else:
                    override_cfg[key] = value
            override_cfg.pop("class_overrides", None)
            overrides.append((classes, _build_train_transform(override_cfg, size, mean, std)))

        if overrides:
            return TransformSelector(default_transform, overrides)
        return TransformWrapper(default_transform)

    valid_transforms = [
        A.Resize(height=size, width=size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return TransformWrapper(A.Compose(valid_transforms))


def _build_train_transform(aug_cfg: Dict[str, Any], size: int, mean, std) -> A.Compose:
    transforms: List[A.BasicTransform] = [
        A.RandomResizedCrop(
            height=size,
            width=size,
            scale=tuple(aug_cfg.get("random_resized_crop_scale", (0.9, 1.0))),
            ratio=tuple(aug_cfg.get("random_resized_crop_ratio", (0.9, 1.1))),
        ),
    ]

    if aug_cfg.get("random_rotate90", False):
        transforms.append(A.RandomRotate90(p=aug_cfg.get("random_rotate90_prob", 1.0)))

    transforms.append(
        A.Rotate(
            limit=aug_cfg.get("max_rotate", 0),
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
            p=aug_cfg.get("rotate_prob", 0.8),
        )
    )

    transforms.append(
        A.Affine(
            scale=tuple(aug_cfg.get("affine_scale", (0.95, 1.05))),
            translate_percent=tuple(aug_cfg.get("affine_translate", (0.02, 0.02))),
            rotate=None,
            shear=None,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            p=0.5,
        )
    )

    if aug_cfg.get("horizontal_flip", False):
        flip_prob = aug_cfg.get("horizontal_flip_prob", 0.5)
        transforms.append(A.HorizontalFlip(p=flip_prob))

    brightness_limit, contrast_limit = aug_cfg.get("brightness_contrast", (0.0, 0.0))
    if brightness_limit > 0 or contrast_limit > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            )
        )

    # Blur: support legacy blur_prob and new dict config
    blur_prob = aug_cfg.get("blur_prob", 0.0)
    blur_cfg = aug_cfg.get("blur")
    if blur_cfg:
        transforms.append(
            A.GaussianBlur(
                p=float(blur_cfg.get("p", 0.0)),
                blur_limit=tuple(blur_cfg.get("blur_limit", (3, 5))),
                sigma_limit=tuple(blur_cfg.get("sigma_limit", (0.2, 0.8))),
            )
        )
    elif blur_prob > 0:
        transforms.append(A.GaussianBlur(p=blur_prob))

    iso_cfg = aug_cfg.get("iso_noise")
    if iso_cfg:
        color_shift_cfg = iso_cfg.get("color_shift", (0.01, 0.05))
        if not isinstance(color_shift_cfg, (list, tuple)):
            color_shift_cfg = (-float(color_shift_cfg), float(color_shift_cfg))
        intensity_cfg = iso_cfg.get("intensity", (0.1, 0.3))
        if not isinstance(intensity_cfg, (list, tuple)):
            intensity_cfg = (0.0, float(intensity_cfg))
        transforms.append(
            A.ISONoise(
                p=float(iso_cfg.get("p", 0.0)),
                color_shift=tuple(color_shift_cfg),
                intensity=tuple(intensity_cfg),
            )
        )

    gauss_cfg = aug_cfg.get("gauss_noise")
    if gauss_cfg:
        transforms.append(
            A.GaussNoise(
                p=float(gauss_cfg.get("p", 0.0)),
                var_limit=tuple(gauss_cfg.get("var_limit", (5.0, 15.0))),
            )
        )

    motion_cfg = aug_cfg.get("motion_blur")
    if motion_cfg:
        transforms.append(
            A.MotionBlur(
                p=float(motion_cfg.get("p", 0.0)),
                blur_limit=tuple(motion_cfg.get("blur_limit", (3, 5))),
            )
        )

    shadow_cfg = aug_cfg.get("random_shadow")
    if shadow_cfg:
        shadow_roi = tuple(shadow_cfg.get("shadow_roi", (0, 0, 1, 1)))
        shadow_kwargs = dict(
            p=float(shadow_cfg.get("p", 0.0)),
            shadow_roi=shadow_roi,
            num_shadows_lower=int(shadow_cfg.get("num_shadows_lower", 1)),
            num_shadows_upper=int(shadow_cfg.get("num_shadows_upper", 2)),
            shadow_dimension=int(shadow_cfg.get("shadow_dimension", 5)),
        )
        transforms.append(A.RandomShadow(**shadow_kwargs))

    clahe_cfg = aug_cfg.get("clahe")
    if clahe_cfg:
        transforms.append(
            A.CLAHE(
                clip_limit=float(clahe_cfg.get("clip_limit", 2.0)),
                tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", (8, 8))),
                p=float(clahe_cfg.get("p", 0.0)),
            )
        )

    coarse_cfg = aug_cfg.get("coarse_dropout")
    if coarse_cfg:
        rel = float(coarse_cfg.get("size", 0.05))
        max_dim = max(1, int(rel * size))
        min_rel = float(coarse_cfg.get("min_size_ratio", rel * 0.5))
        min_dim = max(1, int(min_rel * size))
        transforms.append(
            A.CoarseDropout(
                p=float(coarse_cfg.get("p", 0.0)),
                max_holes=int(coarse_cfg.get("max_holes", 1)),
                max_height=int(coarse_cfg.get("max_height", max_dim)),
                max_width=int(coarse_cfg.get("max_width", max_dim)),
                min_holes=1,
                min_height=int(coarse_cfg.get("min_height", min_dim)),
                min_width=int(coarse_cfg.get("min_width", min_dim)),
                fill_value=coarse_cfg.get("fill_value", 0),
            )
        )

    compression_cfg = aug_cfg.get("compression")
    if compression_cfg:
        transforms.append(
            A.ImageCompression(
                p=float(compression_cfg.get("p", 0.0)),
                quality_lower=int(compression_cfg.get("quality_lower", 60)),
                quality_upper=int(compression_cfg.get("quality_upper", 90)),
            )
        )

    perspective_cfg = aug_cfg.get("perspective")
    if perspective_cfg:
        transforms.append(
            A.Perspective(
                p=float(perspective_cfg.get("p", 0.0)),
                scale=tuple(perspective_cfg.get("scale", (0.01, 0.05))),
                keep_size=True,
            )
        )

    piecewise_cfg = aug_cfg.get("piecewise_affine")
    if piecewise_cfg:
        transforms.append(
            A.PiecewiseAffine(
                p=float(piecewise_cfg.get("p", 0.0)),
                scale=float(piecewise_cfg.get("scale", 0.01)),
            )
        )

    color_jitter_cfg = aug_cfg.get("color_jitter")
    if color_jitter_cfg:
        transforms.append(
            A.ColorJitter(
                p=float(color_jitter_cfg.get("p", 0.0)),
                brightness=float(color_jitter_cfg.get("brightness", 0.2)),
                contrast=float(color_jitter_cfg.get("contrast", 0.2)),
                saturation=float(color_jitter_cfg.get("saturation", 0.2)),
                hue=float(color_jitter_cfg.get("hue", 0.02)),
            )
        )

    rgb_shift_cfg = aug_cfg.get("rgb_shift")
    if rgb_shift_cfg:
        transforms.append(
            A.RGBShift(
                p=float(rgb_shift_cfg.get("p", 0.0)),
                r_shift_limit=int(rgb_shift_cfg.get("r_shift_limit", 10)),
                g_shift_limit=int(rgb_shift_cfg.get("g_shift_limit", 10)),
                b_shift_limit=int(rgb_shift_cfg.get("b_shift_limit", 10)),
            )
        )

    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def _build_reference_transform(aug_cfg: Dict[str, Any], size: int, mean, std) -> A.Compose:
    longest_size = max(size, size)
    divide_transforms = A.OneOf(
        [
            QuarterDivide(p=0.2),
            HalfDivide(p=0.2),
            DivideThreeParts(p=0.2),
            DivideSixParts(p=0.2),
            A.RandomCrop(height=size // 2, width=size // 2, p=0.2),
        ],
        p=0.55,
    )

    strong_geom = A.OneOf(
        [
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(
                shift_limit_x=(-0.2, 0.2),
                shift_limit_y=(-0.2, 0.2),
                scale_limit=(-0.05, 0.05),
                rotate_limit=(-60, 60),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
                rotate_method="largest_box",
                p=0.3,
            ),
            A.Affine(
                scale=(1.0, 1.6),
                translate_percent=None,
                rotate=(-45, 45),
                shear=None,
                keep_ratio=True,
                fit_output=False,
                mode=cv2.BORDER_CONSTANT,
                cval=(255, 255, 255),
                p=0.3,
            ),
            A.OpticalDistortion(
                distort_limit=(-0.3, 0.3),
                shift_limit=(-0.05, 0.05),
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
                p=0.2,
            ),
        ],
        p=0.6,
    )

    ref_transforms: List[A.BasicTransform] = [
        A.Compose(
            [
                A.LongestMaxSize(max_size=longest_size, p=1.0),
                A.PadIfNeeded(
                    min_height=size,
                    min_width=size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                    p=1.0,
                ),
                divide_transforms,
            ],
            p=0.6,
        ),
        A.OneOf(
            [
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Transpose(p=0.4),
            ],
            p=0.6,
        ),
        strong_geom,
        A.Resize(height=size, width=size, always_apply=True, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(ref_transforms)
