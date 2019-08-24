from typing import Union
import numpy as np
from scipy.sparse.csr import csr_matrix


class FrameStack(object):
    def __init__(self, frames: Union[np.ndarray, csr_matrix]):
        self._data = vectorize_data(frames)
        self.as_csr_matrix = isinstance(self._data, csr_matrix)

    @property
    def data(self):
        return self._data


def vectorize_data(frames: Union[np.ndarray, csr_matrix]):
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




