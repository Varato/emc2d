from typing import Tuple, Iterable
import functools
import numpy as np
from .drift_setup import DriftSetup
from .image import Image
from .model import Model, ExpandedModel


def make_crop_boxes(drift_setup: DriftSetup, drift_indices: Iterable[int]) -> Iterable[Tuple[int, int, int, int]]:
    s = drift_setup.image_shape
    boxes = map(lambda pos: (pos[0], pos[0]+s[0], pos[1], pos[1]+s[1]),
                [drift_setup.drift_table[i] for i in drift_indices])
    return boxes


def crop(image: Image, box: Tuple[int, int, int, int]) -> Image:
    return image[box[0]:box[1], box[2]:box[3]]


def expand(model: Model, drift_indices: Iterable[int]) -> ExpandedModel:
    boxes = make_crop_boxes(model.drift_setup, drift_indices)
    crop_this = functools.partial(crop, model.content)
    return ExpandedModel(patches=map(crop_this, boxes),
                         drift_setup=model.drift_setup,
                         drift_indices=drift_indices)


def compose(expanded_model: ExpandedModel) -> Model:
    canvas = np.zeros(shape=expanded_model.model_shape)
    weights = np.zeros(shape=expanded_model.model_shape)
    boxes = make_crop_boxes(expanded_model.drift_setup, expanded_model.drift_indices)
    for patch, box in zip(expanded_model.patches, boxes):
        canvas[box[0]:box[1], box[2]:box[3]] += patch
        weights[box[0]:box[1], box[2]:box[3]] += 1
    canvas /= np.where(weights > 0, weights, 1)
    return Model(canvas, expanded_model.max_drift, expanded_model.image_shape)
