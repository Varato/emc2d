from typing import Tuple, Union, Sequence, Iterator, Iterable
import math
import numpy as np

from .utils import DriftSetup
from .image import Image, StackIter


class Model(Image):
    def __new__(cls,
                init_model: Union[str, None, np.ndarray],
                max_drift: Tuple[int, int],
                image_shape: Tuple[int, int]):
        drift_setup = DriftSetup(max_drift, image_shape)
        init_array = Model._initialize(init_model, drift_setup.model_shape)
        obj = super(Model, cls).__new__(cls, input_array=init_array)
        obj._drift_setup = drift_setup
        return obj

    def __array_finalize__(self, viewed):
        super(Model, self).__array_finalize__(viewed)
        if viewed is None:
            return
        self._drift_setup = getattr(viewed, "_drift_setup", None)

    # Delegate DriftSetup
    @property
    def drift_table(self): return self._drift_setup.drift_table

    @property
    def image_shape(self): return self._drift_setup.image_shape

    @property
    def model_shape(self): return self.shape

    def expand(self, drift_indices: Sequence[int]) -> Iterator[np.ndarray]:
        boxes = self._drift_setup.make_crop_boxes(drift_indices)
        return ExpandedModel(images=map(self.crop, boxes),
                             drift_setup=self._drift_setup,
                             drift_indices=drift_indices)

    @staticmethod
    def _initialize(init_model: Union[str, None, np.ndarray], desired_size: Tuple[int, int]) -> np.ndarray:
        if not isinstance(init_model, np.ndarray):
            if init_model == "random" or init_model is None:
                return np.random.rand(*desired_size)
            else:
                raise ValueError("Unknown init_model type: init_model should be set to 'random', or a numpy array.")
        elif isinstance(init_model, np.ndarray):
            if not init_model.ndim == 2:
                raise ValueError("Invalid init_model shape: it has to be a 2D array.")
            given_size = init_model.shape
            is_smaller = list(lx < ly for lx, ly in zip(given_size, desired_size))
            if any(is_smaller):
                p0 = desired_size[0] - given_size[0] if is_smaller[0] else 0
                p1 = desired_size[1] - given_size[1] if is_smaller[1] else 0
                pad_width = ((math.floor(p0/2), math.ceil(p0/2)), (math.floor(p1/2), math.ceil(p1/2)))
                # TODO: need to think about how to normalize when padding with 0s.
                return np.pad(init_model, pad_width, mode='constant', constant_values=0)
            else:
                return init_model


class ExpandedModel(StackIter):
    def __init__(self,
                 images: Iterable[Image],
                 drift_setup: DriftSetup,
                 drift_indices: Sequence[int]):
        super(ExpandedModel, self).__init__(images)
        self._drift_setup = drift_setup
        self._drift_indices = drift_indices

    def __len__(self):
        return len(self._drift_indices)

    # Delegate DriftSetup
    @property
    def drift_table(self): return self._drift_setup.drift_table

    @property
    def image_shape(self): return self._drift_setup.image_shape

    @property
    def model_shape(self): return self._drift_setup.model_shape

    def compose(self):
        canvas = np.zeros(shape=self.model_shape)
        weights = np.zeros(shape=self.model_shape)
        s = self.image_shape

        for i in self._drift_indices:
            pos = self.drift_table[i]
            canvas[pos[0]:pos[0]+s[0], pos[1]:pos[1]+s[1]] += next(self)
            weights[pos[0]:pos[0]+s[0], pos[1]:pos[1]+s[1]] += 1
        canvas /= np.where(weights > 0, weights, 1)

        return Model(canvas, )

