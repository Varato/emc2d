import numpy as np
from typing import Tuple, Union
import logging
from scipy.sparse import csr_matrix


logger = logging.getLogger("emc2d.utils")


def make_drift_vectors(max_drift: int, origin: str = 'center') -> np.ndarray:
    vectors = np.array([(x, y) for x in range(2*max_drift + 1) for y in range(2*max_drift + 1)], dtype=np.int)

    if origin == 'center':
        return np.array([max_drift, max_drift], dtype=np.int) - vectors
    elif origin == 'corner':
        return vectors
    else:
        raise ValueError("origin must be either 'center' or 'corner'")


def get_translation_series(intensity, window_size, translations) -> np.ndarray:
    n_drifts = translations.shape[0]

    expanded_model = np.empty(shape=(n_drifts, *window_size), dtype=np.float)
    for j, s in enumerate(translations):
        expanded_model[j] = intensity[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]]

    return expanded_model


def normalize(img, mean):
    return img * mean / img.mean()


def vectorize_data(frames: Union[np.ndarray, csr_matrix]):
    """
    initialized the data.
    Parameters
    ----------
    frames: numpy array
        shape: (_num_imgs, *_img_size)
    Notes
    -----
    It checks if the "dense_data" is sparse enough, then decides whether csr sparse format should be used or not.
    If "dense_data" is not very sparse, then overhead of scipy.sparse slows down the procedure "maximize",
    while the data size is huge (e.g. 20000 images), the speed-up of using csr format can be remarkable.
    """
    if isinstance(frames, csr_matrix):
        if frames.ndim == 2:
            return frames
        else:
            raise NotImplementedError("csr_matrix only supports 2D array")
    elif isinstance(frames, np.ndarray):
        n_frames = frames.shape[0]
        vec_data = frames.reshape(n_frames, -1)
        nnz = np.count_nonzero(vec_data)
        data_size = vec_data.shape[0] * vec_data.shape[1]
        ratio = nnz / data_size
        if ratio < 0.01:
            logger.info(f"nnz / data_size = {100 * ratio:.2f}%, using csr sparse data format")
            return csr_matrix(vec_data)
        else:
            logger.info(f"nnz / data_size = {100 * ratio:.2f}%, using dense data format")
            return vec_data


def model_reshape(model: np.ndarray, expected_shape: Tuple[int, int]):
    """
    Pad or crop the model so that its shape becomes expected_shape.

    Parameters
    ----------
    model: 2D array
    expected_shape: Tuple[int ,int]

    Returns
    -------
    the model with expected shape.

    """
    init_shape = model.shape

    # if any dimension of the given model is smaller than the expected shape, pad that dimension.
    is_smaller = [s < st for s, st in zip(init_shape, expected_shape)]
    if any(is_smaller):
        px = expected_shape[0] - init_shape[0] if is_smaller[0] else 0
        py = expected_shape[1] - init_shape[1] if is_smaller[1] else 0
        pad_width = (
            (px // 2, px // 2) if px % 2 == 0 else (px // 2 + 1, px // 2),
            (py // 2, py // 2) if py % 2 == 0 else (py // 2 + 1, py // 2))
        return np.pad(model, pad_width, mode='constant', constant_values=0)
    # if both dimensions of the given model is larger than or equal to the target size, crop it.
    else:
        margin = [init_shape[i] - expected_shape[i] for i in range(2)]
        start_x = margin[0] // 2 if margin[0] % 2 == 0 else margin[0] // 2 + 1
        start_y = margin[1] // 2 if margin[1] % 2 == 0 else margin[1] // 2 + 1
        return model[start_x:start_x + expected_shape[0], start_y:start_y + expected_shape[1]]
