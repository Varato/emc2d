from typing import Tuple, Iterable, Union
import functools
import numpy as np
import numpy.ma as ma
from scipy.sparse.csr import csr_matrix

from .drift_setup import DriftSetup
from .model import Model, ExpandedModel
from .frame_stack import FrameStack


def expand(model: Model, drift_indices: Iterable[int]) -> ExpandedModel:
    boxes = make_crop_boxes(model.drift_setup, drift_indices)
    crop_this = functools.partial(crop, model.content)
    return ExpandedModel(patterns=map(crop_this, boxes),
                         drift_setup=model.drift_setup,
                         drift_indices=drift_indices)


def maximize(expanded_model: ExpandedModel, frame_stack: FrameStack, prior=None) -> ExpandedModel:
    n = len(expanded_model.drift_indices)
    patterns = np.array(list(expanded_model.patterns)).reshape(n, -1)
    membership = membership_probabilities(patterns, frame_stack.data, prior)
    weights = membership / np.sum(membership, axis=1, keepdims=True)
    new_patterns = (weights @ frame_stack.data).reshape(n, *expanded_model.image_shape)
    return ExpandedModel(patterns=new_patterns,
                         drift_setup=expanded_model.drift_setup,
                         drift_indices=expanded_model.drift_indices)


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


def membership_probabilities(patterns: np.ndarray, frames: Union[np.ndarray, csr_matrix], prior=None):
    """
    This function associates membership probabilities of patterns to each frame.
    A prior distribution of patterns (positions) can be given.
    """
    log_pattern = ma.log(patterns).filled(-300)
    log_r = log_pattern @ frames.T - np.sum(patterns, axis=1, keepdims=True)
    log_r_cap = np.max(log_r, axis=0, keepdims=True)
    r = np.exp(np.clip(log_r - log_r_cap, -300.0, 0.0))

    if prior is None:
        p = r / np.sum(r, axis=0, keepdims=True)
    else:
        wr = r * prior.reshape(-1, 1)
        p = wr / np.sum(wr, axis=0, keepdims=True)
    return p

def membership_probabilities2(patterns, frames, prior=None):
    log_r = [[log_likelihood(pattern, frame) for frame in frames] for pattern in patterns]


def log_likelihood(pattern, frame):
    return np.sum(ma.log(pattern).filled(-300) * frame - pattern)


def make_crop_boxes(drift_setup: DriftSetup, drift_indices: Iterable[int]) -> Iterable[Tuple[int, int, int, int]]:
    s = drift_setup.image_shape
    boxes = map(lambda pos: (pos[0], pos[0]+s[0], pos[1], pos[1]+s[1]),
                [drift_setup.drift_table[i] for i in drift_indices])
    return boxes


def crop(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    return image[box[0]:box[1], box[2]:box[3]]


# not pure
def _patch(image: np.ndarray, box: Tuple[int, int, int, int], pattern) -> np.ndarray:
    image[box[0]:box[1], box[2]:box[3]] += pattern
    return image


