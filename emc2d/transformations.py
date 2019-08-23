from typing import Tuple, Iterable
import functools
import numpy as np

from .drift_setup import DriftSetup
from .model import Model, ExpandedModel
from .frame_stack import FrameStack


def make_crop_boxes(drift_setup: DriftSetup, drift_indices: Iterable[int]) -> Iterable[Tuple[int, int, int, int]]:
    s = drift_setup.image_shape
    boxes = map(lambda pos: (pos[0], pos[0]+s[0], pos[1], pos[1]+s[1]),
                [drift_setup.drift_table[i] for i in drift_indices])
    return boxes


def crop(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    return image[box[0]:box[1], box[2]:box[3]]


# mute
def _patch(image: np.ndarray, box: Tuple[int, int, int, int], pattern) -> np.ndarray:
    image[box[0]:box[1], box[2]:box[3]] += pattern
    return image


def expand(model: Model, drift_indices: Iterable[int]) -> ExpandedModel:
    boxes = make_crop_boxes(model.drift_setup, drift_indices)
    crop_this = functools.partial(crop, model.content)
    return ExpandedModel(patterns=map(crop_this, boxes),
                         drift_setup=model.drift_setup,
                         drift_indices=drift_indices)


def compose(expanded_model: ExpandedModel) -> Model:
    boxes = make_crop_boxes(expanded_model.drift_setup, expanded_model.drift_indices)
    canvas, weights = functools.reduce(
        lambda cw, bp: (_patch(cw[0], bp[0], bp[1]), _patch(cw[1], bp[0], 1)),
        zip(boxes, expanded_model.patterns),
        (np.zeros(expanded_model.model_shape), np.zeros(expanded_model.model_shape))
    )
    model_content = canvas / np.where(weights > 0, weights, 1)
    return Model(model_content, expanded_model.max_drift, expanded_model.image_shape)
    
    # boxes = make_crop_boxes(expanded_model.drift_setup, expanded_model.drift_indices)
    # canvas = np.zeros(shape=expanded_model.model_shape)
    # weights = np.zeros(shape=expanded_model.model_shape)
    # for pattern, box in zip(expanded_model.patterns, boxes):
    #     canvas[box[0]:box[1], box[2]:box[3]] += pattern
    #     weights[box[0]:box[1], box[2]:box[3]] += 1
    # canvas /= np.where(weights > 0, weights, 1)
    # return Model(canvas, expanded_model.max_drift, expanded_model.image_shape)


def maximize(expanded_model: ExpandedModel, frames: FrameStack):
    for pattern in expanded_model.patterns:
        pass