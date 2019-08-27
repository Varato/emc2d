"""
This module provides functions that are responsible to the maximization step.
Real-world frames are involved and a lot numeric concerns should be here.
"""
from typing import Union, Iterable, Optional, Tuple, Callable, NewType
from functools import partial
import numpy as np
from scipy.sparse.csr import csr_matrix

from .transforms import expand, compress
from .drift_setup import DriftSetup
from .frame_stack import FrameStack
from .model import Model

Membership = NewType("Memberships", np.ndarray)


def update_model_and_memberships(frame_stack: FrameStack,
                                 drift_setup: DriftSetup,
                                 drift_indices: Optional[Iterable[int]] = None,
                                 prior: Optional[np.ndarray] = None) \
        -> Callable[[Tuple[Model, np.ndarray]], Tuple[Model, np.ndarray]]:

    aggregate_ = partial(aggregate, frame_stack=frame_stack, drift_setup=drift_setup, drift_indices=drift_indices)
    assign_ = partial(assign_memberships, frame_stack=frame_stack, drift_indices=drift_indices, prior=prior)

    def f(model_p_tuple: Tuple[Model, np.ndarray]) -> Tuple[Model, np.ndarray]:
        m, p = model_p_tuple
        m1 = aggregate_(p)
        p1 = assign_(m1)
        return m1, p1
    return f


def aggregate(memberships: np.ndarray,
              frame_stack: FrameStack,
              drift_setup: DriftSetup,
              drift_indices: Optional[Iterable[int]] = None) -> Model:
    """
    This function aggregates frames into patterns according to membership probability

    Parameters
    ----------
    frame_stack: FrameStack object
    drift_setup: DriftSetup object
        It contains information of compatible max_drift, image_shape, model_shape.
    drift_indices: Iterable[int]
        Specifies a subset of the whole drifts, within which the algorithm assigns memberships.
    memberships: 2D array
        The membership probabilities in shape (num_patterns, num_frames).

    Return
    ------
    Model object: the aggregated model.
    """
    n = memberships.shape[0]  # number of patterns (drifts)
    weights = memberships / np.sum(memberships, axis=1, keepdims=True)
    patterns = (weights @ frame_stack.vdata).reshape(n, *drift_setup.image_shape)
    return compress(patterns, drift_setup, drift_indices)


def assign_memberships(model: Model,
                       frame_stack: FrameStack,
                       prior: Optional[np.ndarray] = None,
                       drift_indices: Optional[Iterable[int]] = None) -> np.ndarray:
    """
    Given a model, this function assigns (probabilistic) memberships for each frame in the frame_stack
    within a subset of all possible drifts specifying by drift_indices.

    Parameters
    ----------
    frame_stack: FrameStack object
    drift_indices: Iterable[int]
        Specifies a subset of the whole drifts, within which the algorithm assigns memberships.
    prior: Optional[np.ndarray]
        The prior distribution of drifts
    model: Model object
        The model whose expanded patterns are used to assign the membership probabilities.


    """
    patterns = (np.reshape(p, -1) for p in expand(model, drift_indices))
    memberships = _membership_probabilities(patterns, frame_stack.vdata, prior)
    return memberships


def _membership_probabilities(patterns: Iterable[np.ndarray],
                              frames: Union[np.ndarray, csr_matrix],
                              prior: Optional[np.ndarray] = None) -> np.ndarray:
    """
    This function associates membership probabilities of patterns to each frame.
    A prior distribution of patterns (positions) can be given.
    """
    if isinstance(frames, csr_matrix):  # make use of the faster csr multiplication
        patterns = np.vstack(patterns)
        log_pattern = np.log(patterns)
        log_r = log_pattern @ frames.T - np.sum(patterns, axis=1, keepdims=True)
    else:  # otherwise, keep patterns lazy
        log_r = np.vstack([[_poisson_log_likelihood(pattern, frame) for frame in frames]
                           for pattern in patterns])
    log_r_cap = np.max(log_r, axis=0, keepdims=True)
    r = np.exp(np.clip(log_r - log_r_cap, -300.0, 0.0))

    if prior is None:
        p = r / np.sum(r, axis=0, keepdims=True)
    else:
        w = np.array(prior).reshape(-1, 1)
        if w.shape[0] != r.shape[0]:
            raise ValueError("prior ")
        wr = w * r
        p = wr / np.sum(wr, axis=0, keepdims=True)
    return p


def _poisson_log_likelihood(pattern: np.ndarray, frame: Union[np.ndarray, csr_matrix]):
    return np.sum(np.log(pattern) * frame - pattern)




# def emc_iterator(model, frame_stack, prior):
#     f = partial(aggregate, frame_stack=frame_stack, drift_setup=model.drift_setup)
#     g = partial(assign_memberships, frame_stack=frame_stack, prior=prior)
#     while True:
#         model2 = yield model
#         drift_indices = yield
#         memberships = g(model=model if model2 is None else model2, drift_indices=drift_indices)
#         memberships2 = yield memberships
#         model = f(memberships=memberships if memberships2 is None else memberships2, drift_indices=drift_indices)
