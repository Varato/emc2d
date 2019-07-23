from typing import Tuple, Iterable
from typing import Sequence


class DriftSetup(object):
    def __init__(self, max_drift: Tuple[int, int], image_shape: Tuple[int, int]):
        """
        Parameters
        ----------
            max_drift: Tuple[int, int]
                the absolute value of maximum drift relative to the center of a model.
                It's in unit of pixels.
            image_shape: Tuple[int, int]
                the size of images being drift-corrected
        """
        self._max_drift = max_drift
        self._image_shape = image_shape
        self._model_shape = (
            image_shape[0] + 2 * max_drift[0],
            image_shape[1] + 2 * max_drift[1])

        # drift dimensions in x and y
        self._drift_dim = tuple(2*m+1 for m in self._max_drift)
        # drift table of 2d int vectors
        self._drift_table = DriftSetup.make_drift_table(self._max_drift)

    @property
    def drift_table(self) -> Tuple[Tuple[int, int], ...]:
        return self._drift_table
        
    @property
    def model_shape(self) -> Tuple[int, int]:
        return self._model_shape

    @property
    def image_shape(self) -> Tuple[int, int]:
        return self._image_shape

    @staticmethod
    def make_drift_table(max_drift: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """
        generates all possible drifts given max_drift.
        """
        drift_table = tuple((x,y) for x in range(0, 2 * max_drift[0] + 1) for y in range(0, 2 * max_drift[1] + 1))
        return drift_table

    def get_drift_indices(self, drifts: Sequence[Tuple[int, int]]) -> Sequence[int]:
        return [(x % self._drift_dim[0]) * self._drift_dim[1] + y % self._drift_dim[1] for x, y in drifts]

    def make_crop_boxes(self, drift_indices: Sequence[int]) -> Iterable[Tuple[int, int, int, int]]:
        s = self.image_shape
        boxes = map(lambda pos: (pos[0], pos[0] + s[0], pos[1], pos[1] + s[1]),
                    [self.drift_table[i] for i in drift_indices])
        return boxes
