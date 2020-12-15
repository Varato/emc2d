import time
import numpy as np
from typing import Tuple, Union, List, Optional
from scipy.sparse import csr_matrix

import logging

from .utils import vectorize_data, model_reshape
from .transform import ECOperator
from .calibrate import maximum_likelihood_drifts, centre_by_reference, centre_by_first_frame

_EMC_KERNEL_INSTALLED = True
try:
    from .extensions import emc_kernel
except ImportError:
    _EMC_KERNEL_INSTALLED = False
    print("cpp extension is not correctly installed")


logger = logging.getLogger('emc2d')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class EMC(object):
    def __init__(self,
                 frames: Union[np.ndarray, csr_matrix],
                 frame_size: Tuple[int, int],
                 max_drift: Tuple[int, int],
                 init_model: Union[str, np.ndarray] = 'sum'):
        """
        Parameters
        ----------
        frames: array in shape (num_frames, h, w) or (num_frames, h*w)
        max_drift: Tuple[int, int]
        init_model: str or ndarray
            If it's a string, it should be either 'sum' or 'random'
        """
        self.frames = vectorize_data(frames).astype(np.float32)  # (num_frames, n_pix)
        self.num_frames = frames.shape[0]
        self.frame_size = frame_size
        self.max_drift = max_drift

        # model_size - frame_size = 2*max_drift
        self.model_size = (self.frame_size[0] + 2*self.max_drift[0],
                           self.frame_size[1] + 2*self.max_drift[1])

        # initialize model and assure its size is correct
        model = self.initialize_model(init_model).astype(np.float32)
        self.curr_model = model_reshape(model, self.model_size)

        # the operator for 'expand' and 'compress'
        self.ec_op = ECOperator(max_drift)

        # use this property to select a subset of drifts to be considered.
        # default: consider all drifts.
        self.drifts_in_use = list(range(len(self.ec_op.all_drifts)))

        # to hold the membership probability matrix
        self.membership_probability = None

    def run(self, iterations: int, verbose=True):
        history = {'model_mean': [], 'convergence': []}
        for i in range(iterations):
            last_model = self.curr_model
            start = time.time()
            self.one_step()
            end = time.time()
            power = self.curr_model.mean()
            convergence = np.mean((last_model - self.curr_model)**2)
            if verbose:
                logger.info(f"iter {i+1} / {iterations}: model mean = {power:.3f}, time used = {end-start:.3f} s")
            history['model_mean'].append(power)
            history['convergence'].append(convergence)
        return history

    def one_step(self):
        self.membership_probability = compute_membership_probability(
            frames_flat=self.frames, 
            model=self.curr_model, 
            frame_size=self.frame_size, 
            max_drift=self.max_drift, 
            drifts_in_use=self.drifts_in_use,
            return_raw=False)

        self.curr_model = merge_frames_into_model(
            frames_flat=self.frames, 
            frame_size=self.frame_size, 
            model_size=self.model_size, 
            membership_probability=self.membership_probability,
            max_drift=self.max_drift,
            drifts_in_use=self.drifts_in_use)

    def initialize_model(self, init_model: Union[str, np.ndarray]):
        """
        regularise the initial model, including pad the initial model according to img_size and max_drift.


        Parameters
        ----------
        init_model: str or numpy array
        Returns
        -------
        the regularized initial model
        """
        expected_model_size = (self.frame_size[0] + 2*self.max_drift[0],
                               self.frame_size[1] + 2*self.max_drift[1])

        if (type(init_model) is str) and init_model == 'random':
            return np.random.rand(*expected_model_size)

        if (type(init_model) is str) and init_model == 'sum':
            model = self.frames.sum(0).reshape(*self.frame_size)
        elif type(init_model) is np.ndarray:
            if not init_model.ndim == 2:
                raise ValueError("initial_model has to be a 2D array.")
            model = init_model
        else:
            raise ValueError("unknown initial model type. initial model can be 'random', 'sum', or a numpy array.")

        assert model is not None
        return model_reshape(model, expected_model_size)

    def maximum_likelihood_drifts(self):
        if self.membership_probability is None:
            raise RuntimeError("EMC must be run before estimating drifts")
        return maximum_likelihood_drifts(self.membership_probability, self.ec_op.all_drifts, self.drifts_in_use)

    def centre_by_first_frame(self):
        frame_positions = self.maximum_likelihood_drifts()
        calibrating_drift, recon_drifts = centre_by_first_frame(frame_positions, self.max_drift, centre_is_origin=True)
        return recon_drifts, np.roll(self.curr_model, shift=calibrating_drift, axis=(-2, -1))

    def centre_by_reference(self, reference, centre_is_origin=True):
        frame_positions = self.maximum_likelihood_drifts()
        calibrating_drift, recon_drifts = centre_by_reference(
            frame_positions, self.max_drift, 
            self.frame_size, self.curr_model, reference, self.drifts_in_use, centre_is_origin)
        return recon_drifts, np.roll(self.curr_model, shift=calibrating_drift, axis=(-2, -1))


def compute_membership_probability_memsaving(
        frames_flat,
        model,
        frame_size: Tuple[int, int],
        max_drift: Tuple[int, int], 
        drifts_in_use: Optional[List[int]] = None, 
        return_raw: bool = False):
    """
    Computes the membership probability matrix given expanded_model and frames.

    Parameters
    ----------
    frames_flat: array of shape (N, n_pix)
    model: array of shape (H, W)
    frame_size: Tuple[int, int]
    max_drift: Tuple[int, int]
    drifts_in_use: Optional[List[int]]
        indices that specify what locations in the drift space will be considered
    return_raw: bool
        determines whether to return the reduced log likelihood map directly or not.
        If set False, the log likelihood will be exponentiated and normalized for each frame over all positions.

        where M is the number of positions; N is the number of frames; n_pix is the number of pixels of each frame.
        Notice that both expanded_model and frames are flattened.    model: array of shape (H, W)
        Returns
    -------
    array: the membership probability matrix in shape (M, N)
    """

    if not _EMC_KERNEL_INSTALLED:
        raise RuntimeError("need cpp extension to run this function")

    h, w = frame_size
    max_drift_x, max_drift_y = max_drift

    if drifts_in_use is None:
        drifts_in_use = list(range((2*max_drift_x + 1) * (2*max_drift_y + 1)))
    drifts_in_use = np.array(drifts_in_use, dtype=np.uint32)

    ll = emc_kernel.compute_log_likelihood_map(
        frames_flat=frames_flat.astype(np.float32),
        model=model.astype(np.float32),
        h=h, w=w,
        max_drift_y=max_drift_y,
        drifts_in_use=drifts_in_use
    )
    if return_raw:
        return ll
    ll = np.clip(ll - np.max(ll, axis=0, keepdims=True), -600.0, 1.)
    p_jk = np.exp(ll)

    membershipt_probability = p_jk / (p_jk.sum(0) + 1e-13)
    return membershipt_probability


def compute_membership_probability(
        frames_flat,
        model,
        frame_size: Tuple[int, int],
        max_drift: Tuple[int, int], 
        drifts_in_use: Optional[List[int]] = None, 
        return_raw: bool = False):
    """
    Computes the membership probability matrix given expanded_model and frames.

    Parameters
    ----------
    frames_flat: array of shape (N, n_pix)

    model: array of shape (H, W)

    frame_size: Tuple[int, int]

    max_drift: Tuple[int, int]

    drifts_in_use: Optional[List[int]]
        indices that specify what locations in the drift space will be considered

    return_raw: bool
        determines whether to return the reduced log likelihood map directly or not.
        If set False, the log likelihood will be exponentiated and normalized for each frame over all positions.

        where M is the number of positions; N is the number of frames; n_pix is the number of pixels of each frame.
        Notice that both expanded_model and frames are flattened.    model: array of shape (H, W)
        Returns
    -------
    array: the membership probability matrix in shape (M, N)
    """
    if drifts_in_use is None:
        drifts_in_use = list(range((2*max_drift[0] + 1) * (2*max_drift[1] + 1)))

    ec_op = ECOperator(max_drift)
    expanded_model_flat = ec_op.expand(model, frame_size, drifts_in_use, flatten=True)

    #    (M, N)
    ll = frames_flat.dot(np.log(expanded_model_flat.T + 1e-13)).T - expanded_model_flat.sum(1, keepdims=True)
    if return_raw:
        return ll
    ll = np.clip(ll - np.max(ll, axis=0, keepdims=True), -600.0, 1.)
    p_jk = np.exp(ll)

    membershipt_probability = p_jk / (p_jk.sum(0) + 1e-13)
    return membershipt_probability


def merge_frames_into_model(frames_flat, 
                            frame_size: Tuple[int, int], 
                            model_size: Tuple[int, int], 
                            membership_probability: np.ndarray,
                            max_drift: Tuple[int, int],
                            drifts_in_use: Optional[List[int]] = None):
    """
    Update patterns from frames according to the given membership_prabability.

    Parameters
    ----------
    frames: 2D array in shape (N, h*w)
    frame_size: Tuple[int, int]
        the frame shape (h, w).
    model_size: Tuple[int, int]
        the model shape (H, W).
    membership_probability: 2D array in shape (M, N)
        the membership probabilities for each frame against each drift.
    max_drift: Tuple[int, int]
    drifts_in_use: Optional[List[int]]

    Returns
    -------
    the updated patterns in shape (M, *frame_size)
    """
    if drifts_in_use is None:
        drifts_in_use = list(range((2*max_drift[0] + 1) * (2*max_drift[1] + 1)))

    ec_op = ECOperator(max_drift)

    n_drifts = membership_probability.shape[0]
    weights_jk = membership_probability / membership_probability.sum(1, keepdims=True)  # (n_drifts, n_frames)

    new_w_ji = weights_jk @ frames  # (M, N) @ (N, n_pix) = (M, n_pix)
    new_expanded_model = new_w_ji.reshape(n_drifts, *frame_size)  # (M, h, w)

    return ec_op.compress(new_expanded_model, model_size, drifts_in_use)