from typing import Union, Tuple
import numpy as np
from scipy.sparse.csr import csr_matrix

Array = np.ndarray


class FrameStack(object):
    def __init__(self, frames: Union[np.ndarray, csr_matrix], frame_shape: Tuple[int, int]):
        self._data = vectorize_data(frames)
        self._frame_shape = frame_shape
        self.as_csr_matrix = isinstance(self._data, csr_matrix)

    @property
    def vdata(self):
        return self._data

    @property
    def frame_shape(self):
        return self._frame_shape


def vectorize_data(frames: Union[Array, csr_matrix]):
    if isinstance(frames, csr_matrix):
        if frames.ndim != 2:
            raise ValueError("\"frames\" as a csr_matrix must be rank 2")
        return frames
    elif isinstance(frames, np.ndarray):
        n = frames.shape[0]
        if frames.ndim == 2:
            pass
        elif frames.ndim == 3:
            frames = frames.reshape(n, -1)
        else:
            raise ValueError("\"frames\" as a ndarray must be rank 2 or 3")
        nnz = np.count_nonzero(frames)
        data_size = frames.shape[0] * frames.shape[1]
        ratio = nnz / data_size
        if ratio < 0.01:
            return csr_matrix(frames)
        else:
            return frames




