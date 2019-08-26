from typing import Tuple, Iterable, Optional, Callable, Union
from functools import partial

import numpy as np
from toolz.functoolz import compose

from .numerics import aggregate, assign_memberships
from .drift_setup import DriftSetup
from .frame_stack import FrameStack
from .model import initialize, Model
from .fn_tools import iterate


def emc(frame_stack: FrameStack,
        drift_setup: DriftSetup,
        drift_indices: Iterable[int],
        max_drift: Tuple[int, int],
        init_model: Union[str, None, np.ndarray] = None,
        model_operator: Callable = lambda x: x,
        membership_operator: Callable = lambda x: x,
        prior: Optional[np.ndarray] = None):

    # Need to compute membership probability of the initialzed model before entering the main iteration
    model0 = initialize(max_drift, frame_stack.frame_shape, init_model)
    p0 = assign_memberships(model0, frame_stack, drift_indices, prior)

    f = partial(aggregate, frame_stack=frame_stack, drift_setup=drift_setup, drift_indices=drift_indices)
    g = partial(assign_memberships, frame_stack=frame_stack, drift_indices=drift_indices, prior=prior)

    return iterate(
        compose(g, model_operator, f, membership_operator),
        p0
    )

