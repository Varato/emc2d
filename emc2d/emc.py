import time
import numpy as np
from typing import Tuple, Union
from scipy.sparse import csr_matrix

import logging

from .utils import vectorize_data, model_reshape
from .transform import ECOperator
from .calibrate import maximum_likelihood_drifts, centre_by_reference, centre_by_first_frame


logger = logging.getLogger('emc2d')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class EMC(object):
    def __init__(self,
                 frames: Union[np.ndarray, csr_matrix],
                 frame_size: Tuple[int, int],
                 max_drift: int,
                 init_model: Union[str, np.ndarray] = 'sum'):
        """
        Parameters
        ----------
        frames: array in shape (num_frames, h, w) or (num_frames, h*w)
        max_drift: int
        init_model: str or ndarray
            If it's a string, it should be either 'sum' or 'random'
        """
        self.frames = vectorize_data(frames)  # (num_frames, n_pix)
        self.num_frames = frames.shape[0]
        self.frame_size = frame_size
        self.max_drift = max_drift

        # model_size - frame_size = 2*max_drift
        self.model_size = (self.frame_size[0] + 2*self.max_drift,
                           self.frame_size[1] + 2*self.max_drift)

        # initialize model and assure its size is correct
        model = self.initialize_model(init_model)
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
        expanded_model = self.ec_op.expand(self.curr_model, self.frame_size, self.drifts_in_use, flatten=True)
        self.membership_probability = compute_membership_probability(expanded_model, self.frames)
        new_expanded_model = merge_frames_into_model(self.frames, self.frame_size, self.membership_probability)
        self.curr_model = self.ec_op.compress(new_expanded_model, self.model_size, self.drifts_in_use)

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
        expected_model_size = (self.frame_size[0] + 2*self.max_drift,
                               self.frame_size[1] + 2*self.max_drift)

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

    # def expand_memsaving(self) -> np.ndarray:
    #     """
    #     Expands current model into patterns, and compute the membership probabilities for each frame.

    #     It differs from `expand` in the following way: rather than store the full set of patterns, it
    #     computes the membership probabilities on the fly. In this Python implementation, this saves memory
    #     but is time-inefficient.

    #     Parameters
    #     ----------
        
    #     Returns
    #     -------
    #     membership probabilities as a 2D array in shape (n_drifts, n_frames).
    #     """
    #     n_drifts = len(self.drifts_in_use)
    #     window_size = self.frame_size
    #     membership_probability = np.empty(shape=(n_drifts, self.n_frames), dtype=np.float)

    #     for j, idx in enumerate(self.drifts_in_use):
    #         s = self.drifts[idx]
    #         pattern     = self.curr_model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]].reshape(-1,)
    #         log_pattern = np.log(pattern + 1e-17)
    #         LL = log_pattern @ self.frames.T - np.sum(pattern)  # (n_frames,)
    #         LL = np.clip(LL - np.max(LL), -600.0, 1.)
    #         membership_probability[j, :] = np.exp(LL)

    #     membership_probability /= membership_probability.sum(0, keepdims=True)

    #     return membership_probability


# TODO: cpp extension for memsaving
def compute_membership_probability(expanded_model, frames, return_raw=False):
    """
    Computes the membership probability matrix given expanded_model and frames.

    Parameters
    ----------
    expanded_model: array of shape (M, n_pix)
    frames: array of shape (N, n_pix)
        where M is the number of positions; N is the number of frames; n_pix is the number of pixels of each frame.
        Notice that both expanded_model and frames are flattened.
    return_raw: bool
        determines whether to return the reduced log likelihood map directly or not.
        If set False, the log likelihood will be exponentiated and normalized for each frame over all positions.

    Returns
    -------
    array: the membership probability matrix in shape (M, N)
    """

    #    (M, N)
    ll = frames.dot(np.log(expanded_model.T + 1e-13)).T - expanded_model.sum(1, keepdims=True)
    if return_raw:
        return ll
    ll = np.clip(ll - np.max(ll, axis=0, keepdims=True), -600.0, 1.)
    p_jk = np.exp(ll)

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
    weights_jk = membership_probability / membership_probability.sum(1, keepdims=True)  # (n_drifts, n_frames)

    new_w_ji = weights_jk @ frames  # (M, N) @ (N, n_pix) = (M, n_pix)
    new_expanded_model = new_w_ji.reshape(n_drifts, *frame_size)

    return new_expanded_model
