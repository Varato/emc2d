from typing import Tuple, Iterable, Optional, List
from typing import Sequence

Indices = Iterable[int]


class DriftSetup(object):
    def __init__(self,
                 max_drift: Tuple[int, int],
                 image_shape: Tuple[int, int],
                 model_shape: Optional[Tuple[int, int]] = None):
        """
        Parameters
        ----------
            max_drift: Tuple[int, int]
                the absolute value of maximum drift relative to the center of a model.
                It's in unit of pixels.
            image_shape: Tuple[int, int]
                the size of images being drift-corrected
        """
        if model_shape is None:
            self.model_shape = (image_shape[d] + 2 * max_drift[d] for d in (0, 1))
        else:
            shape_not_match = (model_shape[d] != image_shape[d] + 2 * max_drift[d] for d in (0, 1))
            if any(shape_not_match):
                raise ValueError("constraint model_shape[d] == image_shape[d] + 2 * max_drift[d] does not hold")
            self.model_shape = model_shape

        self.max_drift = max_drift
        self.image_shape = image_shape

        # drift dimensions in x and y
        self.drift_dim = tuple(2*m+1 for m in self.max_drift)
        # drift table of 2d int vectors
        self.drift_table = DriftSetup.make_drift_table(self.max_drift)

    @staticmethod
    def make_drift_table(max_drift: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """
        generates all possible drifts given max_drift.
        """
        drift_table = tuple((x, y) for x in range(0, 2 * max_drift[0] + 1) for y in range(0, 2 * max_drift[1] + 1))
        return drift_table

    def get_drift_indices(self, drifts: Sequence[Tuple[int, int]]) -> Sequence[int]:
        return [(x % self.drift_dim[0]) * self.drift_dim[1] + y % self.drift_dim[1] for x, y in drifts]

    def make_crop_boxes(self, drift_indices: Optional[Indices] = None) -> List[Tuple[int, int, int, int]]:
        s = self.image_shape
        if drift_indices is None:
            drift_indices = range(len(self.drift_table))
        boxes = [(pos[0], pos[0] + s[0], pos[1], pos[1] + s[1]) for pos in [self.drift_table[i] for i in drift_indices]]
        return boxes
