"""
This module provides the core transforms in EMC algorithm: expand and compress, regardless of observations (frames).
No numeric concerns should be here.
"""
from typing import Tuple, Iterable, Union, Optional
import functools
import numpy as np

from .drift_setup import DriftSetup
from .model import Model

Indices = Iterable[int]
Array = np.ndarray
ExpandedModel = Iterable[Array]


def expand(model: Model, drift_indices: Optional[Indices] = None) -> ExpandedModel:
    boxes = model.drift_setup.make_crop_boxes(drift_indices)
    crop_this = functools.partial(crop, model.content)
    patterns = map(crop_this, boxes)
    return patterns


def compress(patterns: ExpandedModel,
             drift_setup: DriftSetup,
             drift_indices: Optional[Indices] = None) -> Model:
    boxes = drift_setup.make_crop_boxes(drift_indices)
    canvas, weights = functools.reduce(
        lambda cw, bp: (_patch(cw[0], bp[0], bp[1]), _patch(cw[1], bp[0], 1)),
        zip(boxes, patterns),
        (np.zeros(drift_setup.model_shape), np.zeros(drift_setup.model_shape))
    )
    model_content = canvas / np.where(weights > 0, weights, 1)
    return Model(model_content, drift_setup.max_drift, drift_setup.image_shape)


def crop(image: Array, box: Tuple[int, int, int, int]) -> Array:
    return image[box[0]:box[1], box[2]:box[3]]


# impure
def _patch(canvas: np.ndarray, box: Tuple[int, int, int, int], pattern) -> np.ndarray:
    canvas[box[0]:box[1], box[2]:box[3]] += pattern
    return canvas


