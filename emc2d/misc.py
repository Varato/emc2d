from typing import List, Tuple, Union
import logging
import numpy as np
from scipy.sparse import csr_matrix

from .utils import make_drift_vectors


logger = logging.getLogger('emc2d')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def maximum_likelihood_drifts(membership_prabability, all_drifts, drifts_in_use: List[int]):
    """
    Parameters
    ----------
    membership_probability: array
        The membership probability matrix computed in expectation maximization
        shape: (M, N), where M is the number of all posible positions of drifts and N is the number of frames.
    """
    frame_position_indices = np.argmax(membership_prabability, axis=0)
    frame_positions = np.array([all_drifts[drifts_in_use[i]] for i in frame_position_indices])
    return frame_positions


def calibrate_drifts_with_reference(
    max_drift: int, 
    frame_size: Tuple[int, int], 
    model, 
    reference,
    drifts_in_use: List[int] = None, 
    centre_is_origin=True):

    ec_op = ECOperator(max_drift)
    if drifts_in_use is None:
        drifts_in_use = list(range(ec_op.num_all_drifts))

    num_drifts = len(drifts_in_use)

    expanded_model = ec_op.expand(model, frame_size, drifts_in_use, flatten=True)
    expanded_ref = ec_op.expand(reference, frame_size, drifts_in_use, flatten=True)

    centre_drift_index = max_drift + max_drift * (2*max_drift + 1)
    if centre_drift_index not in drifts_in_use:
        raise RuntimeError("centre drift is not within the EMC's view. Check the EMC.drift_in_use property")
    idx = drifts_in_use.index(centre_drift_index)
    centre_ref = expanded_ref[idx] # (N,)

    n1 = np.linalg.norm(centre_ref)
    n2 = np.linalg.norm(expanded_model, axis=1, keepdims=True)  #(M, 1)

    v1 = centre_ref / n1         #(N,)
    v2 = expanded_model / n2     #(M, N)

    diff = np.linalg.norm(v1[None, :] - v2, axis=1) #(M, )
    recon_drift_centre_index = np.argmin(diff)
    recon_centre_drift = ec_op.all_drifts[drifts_in_use[recon_drift_centre_index]]

    calibrating_shift = np.array([max_drift, max_drift]) - recon_centre_drift 
    return calibrating_shift


def centre_by_first_frame(frame_positions, max_drift: int, centre_is_origin=True):
    first_frame_position = frame_positions[0]
    calibrating_shift = np.array([max_drift, max_drift]) - first_frame_position
    frame_positions += calibrating_shift
    if centre_is_origin:
        frame_positions -= max_drift
    return calibrating_shift, frame_positions


def centre_by_reference(frame_positions, max_drift: int, frame_size: Tuple[int, int], model, reference, drifts_in_use, centre_is_origin=True):
    calibrating_shift = calibrate_drifts_with_reference(max_drift, frame_size, model, reference, drifts_in_use, centre_is_origin)
    frame_positions += calibrating_shift
    if centre_is_origin:
        frame_positions -= max_drift
    return calibrating_shift, frame_positions


class ECOperator:
    def __init__(self, max_drift):
        self.max_drift = max_drift
        self.all_drifts = make_drift_vectors(max_drift, origin='corner')
        self.num_all_drifts = self.all_drifts.shape[0]

    def expand(self, model, window_size: Tuple[int, int], drifts_in_use: List[int] = None, flatten: bool = False):
        if drifts_in_use is None:
            drifts_in_use = list(range(self.num_all_drifts))

        n_drifts = len(drifts_in_use)
        expanded_model = np.empty(shape=(n_drifts, *window_size), dtype=np.float)
        for j, i in enumerate(drifts_in_use):
            s = self.all_drifts[i]
            expanded_model[j] = model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]]
        
        if flatten:
            return expanded_model.reshape(n_drifts, -1)
        else:
            return expanded_model

    def compress(self, expanded_model, model_size: Tuple[int, int], drifts_in_use: List[int] = None):
        """
        Compresses an expaned_model (patterns) into a model according to the corresponding drifts.

        Parameters
        ----------
        expanded_model: 3D array in shape (n_drifts, h, w)
    
        Returns
        -------
        an assembled model as 2D array
        """
        if drifts_in_use is None:
            drifts_in_use = list(range(self.num_all_drifts))

        window_size = expanded_model.shape[-2:]
        model   = np.zeros(shape=model_size, dtype=np.float)
        weights = np.zeros(shape=model_size, dtype=np.float)
        for k, i in enumerate(drifts_in_use):
            s = self.all_drifts[i]
            model  [s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += expanded_model[k]
            weights[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += 1.0
        return model / np.where(weights>0., weights, 1.)


def compute_membership_probability(expanded_model, frames, return_raw=False):
    """
    Computes the membership probability matrix given expanded_model and frames.

    Parameters
    ----------
    expanded_model: array of shape (M, n_pix)
    frames: array of shape (N, n_pix)
        where M is the number of positions; N is the number of frames; n_pix is the number of pixels of each frame. Notice
        that both expanded_model and frames are flattened.
    return_raw: bool
        determines whether to return the reduced log likelihood map directly or not. If set False, the log likelihood will be 
        exponentiated and normalized for each frame over all positions.

    Returns
    -------
    array: the membership probability matrix in shape (M, N)
    """

    #    (M, N)
    LL = frames.dot(np.log(expanded_model.T + 1e-13)).T - expanded_model.sum(1, keepdims=True)
    if return_raw:
        return LL
    LL = np.clip(LL - np.max(LL, axis=0, keepdims=True), -600.0, 1.)
    p_jk = np.exp(LL)

    membershipt_probability = p_jk / p_jk.sum(0)
    return membershipt_probability


def merge_frames_into_model(frames, frame_size: Tuple[int, int], membership_probability: np.ndarray):
    """
    Update patterns from frames according to the given membership_prabability.

    Parameters
    ----------
    frames: 2D array in shape (N, n_pix)
    frame_size: Tuple[int, int]
        the original height and width of the frames before flattened.
    membership_probability: 2D array in shape (M, N)
        the membership probabilities for each frame against each drift.

    Returns
    -------
    the updated patterns in shape (M, *frame_size)
    """

    n_drifts = membership_probability.shape[0]
    weights_jk = membership_probability / membership_probability.sum(1, keepdims=True) # (n_drifts, n_frames)

    new_w_ji = weights_jk @ frames  # (M, N) @ (N, n_pix) = (M, n_pix)
    new_expanded_model = new_w_ji.reshape(n_drifts, *frame_size)

    return new_expanded_model



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
            logger.info(f"nnz / data_size = {100*ratio:.2f}%, using csr sparse data format")
            return csr_matrix(vec_data)
        else:
            logger.info(f"nnz / data_size = {100*ratio:.2f}%, using dense data format")
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
    is_smaller = [l < lt for l, lt in zip(init_shape, expected_shape)]
    if any(is_smaller):
        px = expected_shape[0] - init_shape[0] if is_smaller[0] else 0
        py = expected_shape[1] - init_shape[1] if is_smaller[1] else 0
        pad_width = (
            (px//2, px//2) if px%2 == 0 else (px//2 + 1, px//2), 
            (py//2, py//2) if py%2 == 0 else (py//2 + 1, py//2))
        return np.pad(model, pad_width, mode='constant', constant_values=0)
    # if both dimensions of the given model is larger than or equal to the target size, crop it.
    else:
        margin = [init_shape[i] - expected_shape[i] for i in range(2)]
        start_x = margin[0]//2 if margin[0]%2 == 0 else margin[0]//2 + 1
        start_y = margin[1]//2 if margin[1]%2 == 0 else margin[1]//2 + 1
        return model[start_x:start_x+expected_shape[0], start_y:start_y+expected_shape[1]]
