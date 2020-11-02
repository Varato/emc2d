import numpy as np
from typing import Tuple, Union

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
