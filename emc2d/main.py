from typing import Tuple, Iterable, Optional, Callable, Union, Generator
from functools import partial

import numpy as np

from .numerics import update_model_and_memberships, assign_memberships, emc_coroutine_iterator
from .frame_stack import FrameStack
from .drift_setup import DriftSetup
from .model import initialize, Model

from .fn_tools import iterate

Indices = Iterable[int]
Array = np.ndarray
Memberships = Array


def emc_simple(frame_stack: FrameStack,
               drift_setup: DriftSetup,
               drift_indices: Optional[Indices] = None,
               init_model: Union[str, None, Array] = None,
               prior: Optional[Array] = None) \
        -> Iterable[Tuple[Model, Memberships]]:

    model_start = initialize(drift_setup.max_drift, drift_setup.image_shape, init_model)
    membership_start = assign_memberships(model_start, frame_stack, prior, drift_indices)

    return iterate(
        update_model_and_memberships(frame_stack, drift_setup, drift_indices, prior),
        (model_start, membership_start)
    )


def emc(frame_stack: FrameStack,
        max_drift: Tuple[int, int],
        num_iters: int = 30,
        init_model: Union[str, None, Array] = None,
        prior: Optional[Array] = None,
        process_model: Callable[[Model], Model] = lambda x: None,
        select_drifts: Callable[[Array], Indices] = lambda x: None) \
        -> Generator[Tuple[Model, Memberships], None, None]:

    model_start = initialize(max_drift, frame_stack.frame_shape, init_model)
    it = emc_coroutine_iterator(model_start, frame_stack, prior=prior)

    m = next(it)
    for _ in range(num_iters):
        p = it.send(process_model(m))
        yield m, p
        m = it.send(select_drifts(p))



