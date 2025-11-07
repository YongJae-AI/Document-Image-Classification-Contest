"""
Custom Albumentations transforms for dividing an image into regions
and returning a random crop among those regions.

Adapted to match the divide-style augmentations from the reference pipeline.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


def quarter_divide(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    quads = [
        image[0:cy, 0:cx],
        image[0:cy, cx:w],
        image[cy:h, 0:cx],
        image[cy:h, cx:w],
    ]
    return random.choice(quads)


def half_divide(image: np.ndarray) -> np.ndarray:
    h = image.shape[0]
    cy = h // 2
    halves = [image[0:cy, :], image[cy:h, :]]
    return random.choice(halves)


def divide_three_parts(image: np.ndarray) -> np.ndarray:
    h = image.shape[0]
    part = h // 3
    segments = [
        image[0:part, :],
        image[part : 2 * part, :],
        image[2 * part :, :],
    ]
    return random.choice(segments)


def divide_six_parts(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    block_h = h // 2
    block_w = w // 3
    blocks = [
        image[0:block_h, 0:block_w],
        image[0:block_h, block_w : 2 * block_w],
        image[0:block_h, 2 * block_w : w],
        image[block_h:h, 0:block_w],
        image[block_h:h, block_w : 2 * block_w],
        image[block_h:h, 2 * block_w : w],
    ]
    return random.choice(blocks)


class _BaseDivide(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        raise NotImplementedError


class QuarterDivide(_BaseDivide):
    def apply(self, img, **params):
        return quarter_divide(img)


class HalfDivide(_BaseDivide):
    def apply(self, img, **params):
        return half_divide(img)


class DivideThreeParts(_BaseDivide):
    def apply(self, img, **params):
        return divide_three_parts(img)


class DivideSixParts(_BaseDivide):
    def apply(self, img, **params):
        return divide_six_parts(img)
