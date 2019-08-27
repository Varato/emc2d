"""
This module provides the core transforms in EMC algorithm: expand and compress, regardless of observations (frames).
No numeric concerns should be here.
"""
from typing import Tuple, Iterable, Union, Optional
import functools
import numpy as np

from .drift_setup import DriftSetup
from .model import Model


def expand(model: Model, drift_indices: Optional[Iterable[int]] = None) -> Iterable[np.ndarray]:
    boxes = make_crop_boxes(model.drift_setup, drift_indices)
    crop_this = functools.partial(crop, model.content)
    patterns = map(crop_this, boxes)
    return patterns


def compress(patterns: Iterable[np.ndarray], drift_setup: DriftSetup, drift_indices: Optional[Iterable[int]] = None) -> Model:
    boxes = make_crop_boxes(drift_setup, drift_indices)
    canvas, weights = functools.reduce(
        lambda cw, bp: (_patch(cw[0], bp[0], bp[1]), _patch(cw[1], bp[0], 1)),
        zip(boxes, patterns),
        (np.zeros(drift_setup.model_shape), np.zeros(drift_setup.model_shape))
    )
    model_content = canvas / np.where(weights > 0, weights, 1)
    return Model(model_content, drift_setup.max_drift, drift_setup.image_shape)


def make_crop_boxes(drift_setup: DriftSetup, drift_indices: Optional[Iterable[int]] = None) -> Iterable[Tuple[int, int, int, int]]:
    s0, s1 = drift_setup.image_shape
    if drift_indices is None:
        drift_indices = range(len(drift_setup.drift_table))
    boxes = map(lambda pos: (pos[0], pos[0]+s0, pos[1], pos[1]+s1),
                [drift_setup.drift_table[i] for i in drift_indices])
    return boxes


def crop(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    return image[box[0]:box[1], box[2]:box[3]]


# not pure
def _patch(image: np.ndarray, box: Tuple[int, int, int, int], pattern) -> np.ndarray:
    image[box[0]:box[1], box[2]:box[3]] += pattern
    return image


