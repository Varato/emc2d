import numpy as np

def make_drift_vectors(max_drift: int, origin: str = 'center') -> np.ndarray:
    vectors = np.array([(x, y) for x in range(2*max_drift + 1) for y in range(2*max_drift + 1)], dtype=np.int)

    if origin == 'center':
        return np.array([max_drift, max_drift], dtype=np.int) - vectors
    elif origin == 'corner':
        return vectors
    else:
        raise ValueError("origin must be either 'center' or 'corner'")