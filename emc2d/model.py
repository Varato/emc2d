from typing import Union, Sequence, Iterator, Iterable
import math
import numpy as np

from .utils import DriftSetup



class Model(object):
    def __init__(self,
                 data: Union[str, None, np.ndarray],
                 mean: float,
                 drift_setup: DriftSetup):

        self._drift_setup = drift_setup
        self._data = self._initialze(data)
        self.normalize(mean)

    def normalize(self, mean):
        self._data *= mean / self._data.mean()

    def crop(self, drift_index: int) -> np.ndarray:
        pos = self._drift_setup.drift_table[drift_index]
        s = self._drift_setup.img_size
        return self._data[pos[0]:pos[0]+s[0], pos[1]:pos[1]+s[1]]

    def expand(self, drift_indices: Sequence[int]) -> Iterator[np.ndarray]:
        return map(self.crop, drift_indices)

    def compose(self, expended_model: Iterator[np.ndarray], drift_indices: Iterable[int]):
        canvas = np.zeros(shape=self._drift_setup.model_size)
        weights = np.zeros(shape=self._drift_setup.model_size)
        s = self._drift_setup.img_size

        for i in drift_indices:
            pos = self._drift_setup.drift_table[i]
            canvas[pos[0]:pos[0]+s[0], pos[1]:pos[1]+s[1]] += next(expended_model)
            weights[pos[0]:pos[0]+s[0], pos[1]:pos[1]+s[1]] += 1
        canvas /= np.where(weights > 0, weights, 1)

        print("data updated")
        self._data = canvas

    def _initialze(self, init_model: Union[str, None, np.ndarray]) -> np.ndarray:
        if not isinstance(init_model, np.ndarray):
            if init_model == "random" or init_model is None:
                return np.random.rand(*self._drift_setup.model_size)
            else:
                raise ValueError("Unknown init_model type: init_model should be set to 'random', or a numpy array.")

        elif isinstance(init_model, np.ndarray):
            if not init_model.ndim == 2:
                raise ValueError("Invalid init_model shape: it has to be a 2D array.") 
                           
            given_size = init_model.shape
            desired_size = self._drift_setup.model_size
            is_smaller = list(lx < ly for lx, ly in zip(given_size, desired_size))
            if any(is_smaller):
                p0 = desired_size[0] - given_size[0] if is_smaller[0] else 0
                p1 = desired_size[1] - given_size[1] if is_smaller[1] else 0
                pad_width = ((math.floor(p0/2), math.ceil(p0/2)), (math.floor(p1/2), math.ceil(p1/2)))
                #TODO: need to think about how to normalize when padding with 0s.
                return np.pad(init_model, pad_width, mode='constant', constant_values=0)
            else:
                return init_model
        
class ExpandedModel(object):
    def __init__(self, expanded_model_iter: Iterator[np.ndarray]):
        self._expanded_model_iter = expanded_model_iter

    def __iter__(self):
        return self._expanded_model_iter

    def __next__(self):
        return next(self._expanded_model_iter)