import numpy as np
from typing import Tuple, Union, List, Optional
import logging
from scipy.sparse import csr_matrix


logger = logging.getLogger("emc2d.utils")


def drift_space_coarse_grain(drift_radius: Tuple[int, int], multiple: Tuple[int, int], return_indices=True):
    """
    Coarse grain the drift space by a multiple.
    
    Parameters
    ----------
    drift_radius: Tuple[int, int]
        the drift space radius along the x, y dimensions
    multiple: Tuple[int, int]
        the multiples for coarse graining of the x, y dimensions
    return_indices: bool
        If set true, then returns indices of drifts in the whole drift space. Otherwise returns x and y coordinates.
    
    Returns
    -------
    coarsed_drift_indices: array of 1D
        The solo indices of the coarse-grained locations in the drift space (2 * drift_radius + 1).
    """
    h, w = [2*r + 1 for r in drift_radius]
    x_locs, y_locs = [np.arange(m//2, 2*r+1, m) for r, m in zip(drift_radius, multiple)]
    if return_indices:
        coarsed_drift_indices = x_locs[:, None] * w + y_locs[None, :]
        return coarsed_drift_indices.flatten()
    return x_locs, y_locs
    
    
def drift_indices_to_locations(drift_radius: Tuple[int, int], drift_indices: Optional[List[int]] = None):
    """
    Convert indices of locations in a drift space to x, y locations.
    The origin is at the corner.
    
    Parameters
    ----------
    drift_radius: Tuple[int, int]
        the drift space radius along the x, y dimensions
    drift_indices: : List[int]
        The solo indices of the locations in the drift space (2 * drift_radius + 1).
        
    Returns
    -------
    x: array of shape (n,)
    y: array of shape (n,)
        x, y coordinates of the locations corresponding to the drift_indices, where n = len(drift_indices).
    """
    if drift_indices is None:
        drift_indices = list(range((2*drift_radius[0] + 1) * (2*drift_radius[1] + 1)))

    drift_indices = np.asarray(drift_indices, dtype=np.int)
    h, w = [2*r + 1 for r in drift_radius]
    x = drift_indices // w
    y = drift_indices % w
    return x, y


def fold_likelihood_map(membership_probability,
                        drift_radius: Tuple[int, int],
                        drifts_in_use: Optional[List[int]] = None):
    h, w = [2*r + 1 for r in drift_radius]
    if drifts_in_use is None:
        drifts_in_use = list(range(h*w))

    M, N = membership_probability.shape
    if M != len(drifts_in_use):
        raise ValueError("membership probability dimension does not match the length of the given drifts_in_use")

    xrange, yrange = [np.arange(0, h), np.arange(0, w)]
    x, y = drift_indices_to_locations(drift_radius, drifts_in_use)
    pmat = np.zeros(shape=(N, h, w), dtype=np.float32)
    pmat[:, x, y] = membership_probability.T
    return pmat


def centre_crop(img, size: Tuple[int, int]):
    sx, sy = (size, size) if type(size) is int else size
    mx, my = (img.shape[0] - sx) // 2, (img.shape[1] - sy) // 2
        
    ret = img[mx:mx+sx, my:my+sy]
    return ret


def make_drift_vectors(drift_radius: Tuple[int, int], origin: str = 'center') -> np.ndarray:
    vectors = np.array([(x, y) for x in range(2*drift_radius[0] + 1) for y in range(2*drift_radius[1] + 1)], dtype=np.int)

    if origin == 'center':
        return np.array(drift_radius, dtype=np.int) - vectors
    elif origin == 'corner':
        return vectors
    else:
        raise ValueError("origin must be either 'center' or 'corner'")


def group_frames(frames, group_size: int):
    if group_size == 1:
        return frames
    num_frames = frames.shape[0]
    remainder = num_frames % group_size
    whole = int(num_frames - remainder)

    new_frames = np.array([np.sum(frames[i:i + group_size, :, :], axis=0) for i in range(0, whole, group_size)])
    return new_frames


def group_motions(traj, group_size: int, average: bool = True):
    if group_size == 1:
        return traj

    num_frames = traj.shape[0]
    remainder = num_frames % group_size
    whole = int(num_frames - remainder)

    new_traj = np.array([traj[i:i + group_size, :] for i in range(0, whole, group_size)])
    if average:
        new_traj = np.sum(new_traj, axis=1)/group_size
    return new_traj


def get_spectrum(image, center_cover_size: int = 20, max_normalized: bool = True):
    h, w = image.shape
    c = center_cover_size
    ft = np.fft.fftshift(np.fft.fft2(image))
    ps = np.abs(ft)**2
    ch_start = (h-c)//2
    cw_start = (w-c)//2
    if max_normalized:
        ps /= ps.max()
    ps[ch_start:ch_start+c, cw_start:cw_start+c] = 0
    return ps


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


def model_reshape(model: np.ndarray, expected_shape: Tuple[int, int], return_mask=True):
    """
    Pad or crop the model so that its shape becomes expected_shape.

    Parameters
    ----------
    model: 2D array
    expected_shape: Tuple[int ,int]
    return_mask: bool

    Returns
    -------
    model: array
        the model with expected shape.
    mask: array
        a mask that manifests how the model reshaped.
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
        mask = np.pad(np.ones(init_shape), pad_width, mode='constant', constant_values=0)
        return np.pad(model, pad_width, mode='constant', constant_values=0), mask
    # if both dimensions of the given model is larger than or equal to the target size, crop it.
    else:
        margin = [init_shape[i] - expected_shape[i] for i in range(2)]
        start_x = margin[0] // 2 if margin[0] % 2 == 0 else margin[0] // 2 + 1
        start_y = margin[1] // 2 if margin[1] % 2 == 0 else margin[1] // 2 + 1
        mask = np.ones(init_shape)
        mask[start_x:start_x + expected_shape[0], start_y:start_y + expected_shape[1]] = 0
        return model[start_x:start_x + expected_shape[0], start_y:start_y + expected_shape[1]], mask
