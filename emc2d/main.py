from typing import Tuple, Iterable, Optional, Callable, Union
from functools import partial

import numpy as np

from .numerics import update_model_and_memberships, assign_memberships
from .frame_stack import FrameStack
from .drift_setup import DriftSetup
from .model import initialize, Model

from .fn_tools import iterate


def emc(frame_stack: FrameStack,
        drift_setup: DriftSetup,
        drift_indices: Optional[Iterable[int]] = None,
        init_model: Union[str, None, np.ndarray] = None,
        prior: Optional[np.ndarray] = None) -> Iterable[Tuple[Model, np.ndarray]]:

    model_start = initialize(drift_setup.max_drift, drift_setup.image_shape, init_model)
    membership_start = assign_memberships(model_start, frame_stack, prior, drift_indices)

    return iterate(
        update_model_and_memberships(frame_stack, drift_setup, drift_indices, prior),
        (model_start, membership_start)
    )





