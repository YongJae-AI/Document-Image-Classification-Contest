from typing import Any, Dict

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def create_transforms(cfg: Dict[str, Any], is_train: bool):
    size = cfg["data"]["input_size"]
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]

    if is_train:
        aug_cfg = cfg["augmentations"]
        transforms = [
            A.RandomResizedCrop(
                height=size,
                width=size,
                scale=tuple(aug_cfg.get("random_resized_crop_scale", (0.9, 1.0))),
                ratio=tuple(aug_cfg.get("random_resized_crop_ratio", (0.9, 1.1))),
            ),
        ]

        if aug_cfg.get("random_rotate90", False):
            transforms.append(
                A.RandomRotate90(p=aug_cfg.get("random_rotate90_prob", 1.0))
            )

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

        blur_prob = aug_cfg.get("blur_prob", 0.0)
        if blur_prob > 0:
            transforms.append(A.GaussianBlur(p=blur_prob))

        transforms.extend(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        transforms = [
            A.Resize(height=size, width=size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(transforms)
