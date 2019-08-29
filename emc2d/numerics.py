"""
This module provides functions that are responsible to the maximization step.
Real-world frames are involved and a lot numeric concerns should be here.
"""
from typing import Union, Iterable, Optional, Tuple, Callable, Generator
from functools import partial
import numpy as np
from scipy.sparse.csr import csr_matrix

from .transforms import expand, compress
from .drift_setup import DriftSetup
from .frame_stack import FrameStack
from .model import Model

Indices = Iterable[int]
Array = np.ndarray
Memberships = np.ndarray


def emc_coroutine_iterator(model: Model,
                           frame_stack: FrameStack,
                           drift_indices: Optional[Indices] = None,
                           prior: Optional[Array] = None) \
        -> Generator[Union[Model, Memberships], Union[Model, Indices], None]:

    """
    This iterator provides a co-routine mechanism to user so that the program can interleave between the EMC
    library space and user space.

    In the user space, user could:
        1. modify the model produced by EMC (e.g. low-pass filtering it) and send it back to the main routine.
        2. select a subset from the whole drift space by specifying "drift_indices" and send it to the main routine
           so that the model in the next iteration will only be aggregated from the subset.
    Schematically, this idea is shown as follows.

    this routine:   m0 ---→ p0 -------------→ m1 ---→ p1 ...
    co-routine:      ↘ m0_ ↗ ↘ drift_indices ↗ ↘ m1_ ↗   ...

    where "mi" and "pi" means the model and membership probabilities at i-th iteration respectively; "mi_" means
    the modified model given by the user, which is used to produce "pi".
    """
    aggregate_ = partial(aggregate, frame_stack=frame_stack, drift_setup=model.drift_setup)
    assign_ = partial(assign_memberships, frame_stack=frame_stack, prior=prior)
    while True:
        model_ = yield model
        memberships = assign_(model=model_ if model_ else model, drift_indices=drift_indices)
        drift_indices = yield memberships
        model = aggregate_(memberships=memberships, drift_indices=drift_indices)


def update_model_and_memberships(frame_stack: FrameStack,
                                 drift_setup: DriftSetup,
                                 drift_indices: Optional[Indices] = None,
                                 prior: Optional[Array] = None) \
        -> Callable[[Tuple[Model, Memberships]], Tuple[Model, Memberships]]:

    aggregate_ = partial(aggregate, frame_stack=frame_stack, drift_setup=drift_setup, drift_indices=drift_indices)
    assign_ = partial(assign_memberships, frame_stack=frame_stack, drift_indices=drift_indices, prior=prior)

    def f(model_p_tuple: Tuple[Model, np.ndarray]) -> Tuple[Model, np.ndarray]:
        m, p = model_p_tuple
        m1 = aggregate_(p)
        p1 = assign_(m1)
        return m1, p1
    return f


def aggregate(memberships: Memberships,
              frame_stack: FrameStack,
              drift_setup: DriftSetup,
              drift_indices: Optional[Indices] = None) -> Model:
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
                       prior: Optional[Array] = None,
                       drift_indices: Optional[Indices] = None) -> Memberships:
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


def _membership_probabilities(patterns: Iterable[Array],
                              frames: Union[Array, csr_matrix],
                              prior: Optional[Array] = None) -> Array:
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


def _poisson_log_likelihood(pattern: Array, frame: Union[Array, csr_matrix]):
    return np.sum(np.log(pattern) * frame - pattern)



