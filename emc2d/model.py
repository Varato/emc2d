from typing import Tuple, Union, Iterable
import math
import numpy as np

from .utils import DriftSetup
from .image import Image


class _ModelBase(object):
    def __new__(cls, *args, **kwargs):
        super(_ModelBase, cls).__new__(args, kwargs)

    def __init__(self, drift_setup: DriftSetup):
        self._drift_setup = drift_setup

    # Delegate DriftSetup
    @property
    def drift_table(self):
        return self._drift_setup.drift_table

    @property
    def image_shape(self):
        return self._drift_setup.image_shape

    @property
    def model_shape(self):
        return self._drift_setup.model_shape

    @property
    def max_drift(self):
        return self._drift_setup.max_drift

    @property
    def num_drifts(self):
        return len(self._drift_setup.drift_table)


class Model(_ModelBase):
    def __init__(self, data, max_drift: Tuple[int, int], image_shape: Tuple[int, int]):
        drift_setup = DriftSetup(max_drift, image_shape, data.shape)  # raise exception if shape cannot match
        super(Model, self).__init__(drift_setup)
        self._content = Image(data)

    @property
    def content(self) -> Image:
        return self._content

    def expand(self, drift_indices: Iterable[int]) -> 'ExpandedModel':
        crops = self._drift_setup.make_crop_boxes(drift_indices)
        return ExpandedModel(images=map(self._content.crop, crops),
                             drift_setup=self._drift_setup,
                             drift_indices=drift_indices)


class ExpandedModel(_ModelBase):
    def __init__(self, images: Iterable[Image], drift_indices: Iterable[int], drift_setup: DriftSetup):
        super(ExpandedModel, self).__init__(drift_setup)
        self._drift_indices = drift_indices
        self._patches = images

    @property
    def drift_indices(self): return self._drift_indices

    def compose(self) -> Model:
        canvas = np.zeros(shape=self.model_shape)
        weights = np.zeros(shape=self.model_shape)
        s = self.image_shape
        for patch, index in zip(self._patches, self._drift_indices):
            pos = self.drift_table[index]
            canvas[pos[0]:pos[0] + s[0], pos[1]:pos[1] + s[1]] += patch
            weights[pos[0]:pos[0] + s[0], pos[1]:pos[1] + s[1]] += 1
        canvas /= np.where(weights > 0, weights, 1)
        return Model(canvas, self.max_drift, self.image_shape)


def initialize(max_drift: Tuple[int, int], image_shape: Tuple[int, int],
               init_model: Union[str, None, np.ndarray] = None) -> Model:
    desired_shape = tuple(image_shape[d] + 2 * max_drift[d] for d in (0, 1))
    if not isinstance(init_model, np.ndarray):
        if init_model == "random" or init_model is None:
            return Model(np.random.rand(*desired_shape), max_drift, image_shape)
        else:
            raise ValueError("Unknown init_model type: init_model should be set to 'random', or a numpy array.")
    elif isinstance(init_model, np.ndarray):
        if not init_model.ndim == 2:
            raise ValueError("Invalid init_model shape: it has to be a 2D array.")
        given_size = init_model.shape
        diff = tuple(s0 - s1 for s0, s1 in zip(desired_shape, given_size))
        # pad or cut the given array to desired shape
        # TODO: need to think about how to normalize when padding with 0s.
        modified = init_model
        for d in (0, 1):
            if diff[d] > 0:
                tmp = [(math.floor(diff[d]/2), math.ceil(diff[d]/2)), (0, 0)]
                pad_width = tmp if d == 0 else list(reversed(tmp))
                modified = np.pad(modified, pad_width, mode='constant', constant_values=0)
            elif diff[d] < 0:
                start = (-diff[d]) // 2
                stop = start + desired_shape[d]
                modified = modified[start: stop, :] if d == 0 else modified[:, start: stop]
            else:
                pass
        return Model(modified, max_drift, image_shape)

