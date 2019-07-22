from typing import List, Tuple
from typing import Sequence

class DriftSetup(object):
    def __init__(self, max_drift: Tuple[int, int], img_size: Tuple[int, int]):
        """
        Paremeters
        ----------
            max_drift: Tuple[int, int]
                the absolute value of maximum drift relative to the center of a model.
                It's in unit of pixels.
            img_size: Tuple[int, int]
                the size of images being drift-corrected
        """
        self._max_drift = max_drift
        self._img_size = img_size
        self._model_size = (
            img_size[0] + 2 * max_drift[0],
            img_size[1] + 2 * max_drift[1])

        #drift dimensions in x and y
        self._drift_dim = tuple(2*m+1 for m in self._max_drift)
        #drift table of 2d int vectors
        self._drift_table = DriftSetup.make_drift_table(self._max_drift)

    @property
    def drift_table(self) -> Tuple[Tuple[int, int], ...]:
        return self._drift_table
        
    @property
    def model_size(self) -> Tuple[int, int]:
        return self._model_size

    @property
    def img_size(self) -> Tuple[int, int]:
        return self._img_size

    @staticmethod
    def make_drift_table(max_drift: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """
        generates all possible drifts given max_drift.
        """
        drift_table = tuple((x,y) 
            for x in range(0, 2 * max_drift[0] + 1) 
            for y in range(0, 2 * max_drift[1] + 1))
        return drift_table

    def get_drift_indices(self, drifts: Sequence[Tuple[int, int]]) -> Sequence[int]:
        return [(x % self._drift_dim[0]) * self._drift_dim[1] + y % self._drift_dim[1]
            for x, y in drifts]

# d = DriftSetup(max_drift=(3,3), img_size=(4,4))
# print(d.drift_table)
# print(d.get_drift_indices([(0,0), (6,6), (5,3)]))
