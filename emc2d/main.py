from typing import Tuple, Iterable, Optional, Callable, Union, Generator
from functools import partial

import numpy as np

from .numerics import update_model_and_memberships, assign_memberships, emc_coroutine_iterator
from .frame_stack import FrameStack
from .drift_setup import DriftSetup
from .model import initialize, Model


Indices = Iterable[int]
Array = np.ndarray
Memberships = Array


class Emc:
    def __init__(self,
                 frame_stack: FrameStack,
                 drift_setup: DriftSetup,
                 drift_indices: Optional[Indices] = None,
                 init_model: Union[str, None, Array] = None,
                 prior: Optional[Array] = None):

        self._update_func = update_model_and_memberships(frame_stack, drift_setup, drift_indices, prior)
        self._model = initialize(drift_setup.max_drift, drift_setup.image_shape, init_model)
        self._membership = assign_memberships(self._model, frame_stack, prior, drift_indices)

    def run(self, steps=1):
        for _ in range(steps):
            self._model, self._membership = self._update_func(self._membership)

    @property
    def model(self) -> Array:
        return self._model.content

    @model.setter
    def model(self, m):
        self._model.content = m

    @property
    def membership(self):
        return self._membership


# def emc_simple(frame_stack: FrameStack,
#                drift_setup: DriftSetup,
#                drift_indices: Optional[Indices] = None,
#                init_model: Union[str, None, Array] = None,
#                prior: Optional[Array] = None) \
#         -> Iterable[Tuple[Model, Memberships]]:
#
#     model_start = initialize(drift_setup.max_drift, drift_setup.image_shape, init_model)
#     membership_start = assign_memberships(model_start, frame_stack, prior, drift_indices)
#
#     return iterate(
#         update_model_and_memberships(frame_stack, drift_setup, drift_indices, prior),
#         (model_start, membership_start)
#     )
#
#
# def emc(frame_stack: FrameStack,
#         max_drift: Tuple[int, int],
#         num_iters: int = 30,
#         init_model: Union[str, None, Array] = None,
#         prior: Optional[Array] = None,
#         process_model: Callable[[Model], Model] = lambda x: None,
#         select_drifts: Callable[[Array], Indices] = lambda x: None) \
#         -> Generator[Tuple[Model, Memberships], None, None]:
#
#     model_start = initialize(max_drift, frame_stack.frame_shape, init_model)
#     it = emc_coroutine_iterator(model_start, frame_stack, prior=prior)
#
#     m = next(it)
#     for _ in range(num_iters):
#         p = it.send(process_model(m))
#         yield m, p
#         m = it.send(select_drifts(p))
