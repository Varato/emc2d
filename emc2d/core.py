import warnings
import numpy as np
from typing import Tuple, Union, List, Optional
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter

from tqdm import tqdm

import logging

from .utils import vectorize_data, model_reshape, fold_likelihood_map
from .transform import ECOperator
from .calibrate import maximum_likelihood_drifts, centre_by_reference, centre_by_first_frame

_EMC_KERNEL_INSTALLED = True
try:
    from .extensions import emc_kernel
except ImportError as e:
    _EMC_KERNEL_INSTALLED = False
    warnings.warn(f"cpp extension is not correctly installed: {str(e)}")

logger = logging.getLogger('emc2d')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class EMC(object):
    def __init__(self,
                 frames: Union[np.ndarray, csr_matrix],
                 frame_size: Tuple[int, int],
                 drift_radius: Tuple[int, int],
                 init_model: Union[str, np.ndarray] = 'sum'):
        """
        Parameters
        ----------
        frames: array in shape (num_frames, h, w) or (num_frames, h*w)
        drift_radius: Tuple[int, int]
        init_model: str or ndarray
            If it's a string, it should be either 'sum' or 'random'
        """
        self.frames = vectorize_data(frames).astype(np.float64)  # (num_frames, n_pix)
        self.num_frames = frames.shape[0]
        self.frame_size = frame_size
        self.drift_radius = drift_radius
        self.frames_mean = self.frames.mean()

        # model_size - frame_size = 2*drift_radius
        self.model_size = (self.frame_size[0] + 2*self.drift_radius[0],
                           self.frame_size[1] + 2*self.drift_radius[1])

        # initialize model and assure its size is correct
        self.curr_model = self.initialize_model(init_model, pixel_mean=self.frames_mean).astype(np.float64)

        # the operator for 'expand' and 'compress'
        self.ec_op = ECOperator(drift_radius)

        # to hold the membership probability matrix
        self.membership_probability = None

        self.history = {'model_mean': [], 'convergence': []}

        self._drifts_in_use = None

    @property
    def drifts_in_use(self):
        n_drifts = (2*self.drift_radius[0] + 1) * (2*self.drift_radius[1] + 1)
        if self._drifts_in_use is None:
            return list(range(n_drifts))
        return self._drifts_in_use

    @property
    def folded_membership_probability(self):
        if self.membership_probability is None:
            return None
        return fold_likelihood_map(self.membership_probability, self.drift_radius, self.drifts_in_use)

    def binary_location_map(self, threshold: float = 1e-3):
        pmat = self.folded_membership_probability
        binary = pmat >= threshold
        return binary

    def discard_frames(self, frame_indices: List[int]):
        # TODO: delete for csr_matrix
        np.delete(self.frames, frame_indices, axis=0)

    def run(self, iterations: int, drifts_in_use: List[int] = None, lpfs: float = None):
        for _ in tqdm(range(iterations), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            last_model = self.curr_model
            self.one_step(drifts_in_use, lpfs)
            power = self.curr_model.mean()
            convergence = np.mean((last_model - self.curr_model)**2)
            self.history['model_mean'].append(power)
            self.history['convergence'].append(convergence)

    def run_frame_sparse(self, iterations: int, frame_drifts_in_use: List[List[int]], lpfs: float = None):
        for _ in tqdm(range(iterations), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            last_model = self.curr_model
            self.one_step_frame_sparse(frame_drifts_in_use, lpfs)
            power = self.curr_model.mean()
            convergence = np.mean((last_model - self.curr_model)**2)
            self.history['model_mean'].append(power)
            self.history['convergence'].append(convergence)

    def run_memsaving(self, iterations: int, lpfs: float = None):
        for _ in tqdm(range(iterations), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            last_model = self.curr_model
            self.one_step_memsaving(lpfs)
            power = self.curr_model.mean()
            convergence = np.mean((last_model - self.curr_model)**2)
            self.history['model_mean'].append(power)
            self.history['convergence'].append(convergence)

    def one_step_frame_sparse(self, frames_drifts_in_use: List[List[int]], lpfs: float = None):
        if lpfs is not None:
            self.curr_model = gaussian_filter(self.curr_model, sigma=lpfs)

        self.membership_probability = compute_membership_probability_frame_spase_memsaving(
            frames_flat=self.frames,
            model=self.curr_model,
            frame_size=self.frame_size,
            drift_radius=self.drift_radius,
            frame_drifts_in_use=frames_drifts_in_use)

        self.curr_model = merge_frames_soft(
            frames_flat=self.frames,
            frame_size=self.frame_size,
            model_size=self.model_size,
            membership_probability=self.membership_probability,
            drift_radius=self.drift_radius,
            drifts_in_use=None)

        self._drifts_in_use = None

    def one_step(self, drifts_in_use: List[int] = None, lpfs: float = None):
        if lpfs is not None:
            self.curr_model = gaussian_filter(self.curr_model, sigma=lpfs)

        self.membership_probability = compute_membership_probability(
            frames_flat=self.frames,
            model=self.curr_model,
            frame_size=self.frame_size,
            drift_radius=self.drift_radius,
            drifts_in_use=drifts_in_use,
            return_raw=False)

        self.curr_model = merge_frames_soft(
            frames_flat=self.frames, 
            frame_size=self.frame_size, 
            model_size=self.model_size, 
            membership_probability=self.membership_probability,
            drift_radius=self.drift_radius,
            drifts_in_use=drifts_in_use)

        self._drifts_in_use = drifts_in_use

    def one_step_memsaving(self, lpfs: float = None):
        if lpfs is not None:
            self.curr_model = gaussian_filter(self.curr_model, sigma=lpfs)

        self.membership_probability = compute_membership_probability_memsaving(
            frames_flat=self.frames, 
            model=self.curr_model, 
            frame_size=self.frame_size, 
            drift_radius=self.drift_radius,
            drifts_in_use=self.drifts_in_use,
            return_raw=False)

        self.curr_model = merge_frames_soft_memsaving(
            frames_flat=self.frames, 
            frame_size=self.frame_size, 
            model_size=self.model_size, 
            membership_probability=self.membership_probability,
            drift_radius=self.drift_radius,
            drifts_in_use=self.drifts_in_use)

    def initialize_model(self, init_model: Union[str, np.ndarray], pixel_mean: float):
        """
        regularise the initial model, including pad the initial model according to img_size and drift_radius.


        Parameters
        ----------
        init_model: str or numpy array

        pixel_mean: float

        Returns
        -------
        the regularized initial model
        """

        if (type(init_model) is str) and init_model == 'random':
            model = np.random.rand(*self.model_size)
            model = model * pixel_mean / model.mean()
            return model

        if (type(init_model) is str) and init_model == 'sum':
            model = self.frames.sum(0).reshape(*self.frame_size)
            model = model * pixel_mean / model.mean()
            model, mask = model_reshape(model, self.model_size)
            noise = np.where(mask == 1, 0, np.random.rand(*mask.shape)*pixel_mean*0.5)
            return model + noise

        if type(init_model) is np.ndarray:
            if not init_model.ndim == 2:
                raise ValueError("init_model has to be a 2D array.")
            model, _ = model_reshape(init_model, self.model_size)
            model = model * pixel_mean / model.mean()
            return model
        raise ValueError("unknown initial model type. initial model can be 'random', 'sum', or a numpy array.")

    def maximum_likelihood_drifts(self):
        if self.membership_probability is None:
            raise RuntimeError("EMC must be run before estimating drifts")
        return maximum_likelihood_drifts(self.membership_probability, self.ec_op.all_drifts, self.drifts_in_use)

    def centre_by_first_frame(self):
        frame_positions = self.maximum_likelihood_drifts()
        calibrating_drift, recon_drifts = centre_by_first_frame(
            frame_positions, self.drift_radius, centre_is_origin=True)
        return recon_drifts, np.roll(self.curr_model, shift=calibrating_drift, axis=(-2, -1))

    def centre_by_reference(self, reference, centre_is_origin=True):
        frame_positions = self.maximum_likelihood_drifts()
        calibrating_drift, recon_drifts = centre_by_reference(
            frame_positions, self.drift_radius,
            self.frame_size, self.curr_model, reference, self.drifts_in_use, centre_is_origin)
        return recon_drifts, np.roll(self.curr_model, shift=calibrating_drift, axis=(-2, -1))


def compute_membership_probability_memsaving(
        frames_flat,
        model,
        frame_size: Tuple[int, int],
        drift_radius: Tuple[int, int],
        drifts_in_use: Optional[List[int]] = None,
        return_raw: bool = False):
    """
    Computes the membership probability matrix given expanded_model and frames.

    Parameters
    ----------
    frames_flat: array of shape (N, n_pix)
    model: array of shape (H, W)
    frame_size: Tuple[int, int]
    drift_radius: Tuple[int, int]
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
    max_drift_x, max_drift_y = drift_radius

    if drifts_in_use is None:
        drifts_in_use = list(range((2*max_drift_x + 1) * (2*max_drift_y + 1)))
    drifts_in_use = np.ascontiguousarray(drifts_in_use, dtype=np.uint32)

    ll = emc_kernel.compute_log_likelihood_map(
        frames_flat=frames_flat.astype(np.float32),
        model=model.astype(np.float32),
        h=h, w=w,
        drift_radius_y=max_drift_y,
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
        drift_radius: Tuple[int, int],
        drifts_in_use: Optional[List[int]] = None,
        return_raw: bool = False):
    """
    Computes the membership probability matrix given expanded_model and frames.

    Parameters
    ----------
    frames_flat: array of shape (N, n_pix)

    model: array of shape (H, W)

    frame_size: Tuple[int, int]

    drift_radius: Tuple[int, int]

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
        drifts_in_use = list(range((2*drift_radius[0] + 1) * (2*drift_radius[1] + 1)))

    ec_op = ECOperator(drift_radius)
    expanded_model_flat = ec_op.expand(model, frame_size, drifts_in_use, flatten=True)

    #    (M, N)
    ll = frames_flat.dot(np.log(expanded_model_flat.T + 1e-13)).T - expanded_model_flat.sum(1, keepdims=True)
    if return_raw:
        return ll
    ll = np.clip(ll - np.max(ll, axis=0, keepdims=True), -1000.0, 1.)
    p_jk = np.exp(ll)

    membershipt_probability = p_jk / p_jk.sum(0)
    return membershipt_probability


def compute_membership_probability_frame_spase(frames_flat,
                                               model,
                                               frame_size: Tuple[int, int],
                                               drift_radius: Tuple[int, int],
                                               frame_drifts_in_use: Optional[List[List[int]]]):
    n_frames = frames_flat.shape[0]
    n_drifts = (2*drift_radius[0] + 1) * (2*drift_radius[1] + 1)
    ec_op = ECOperator(drift_radius)

    ll = -np.ones(shape=(n_frames, n_drifts), dtype=np.float64) * np.infty  # (N, M)
    for k in range(n_frames):
        fdidx = frame_drifts_in_use[k]
        wji = ec_op.expand(model, frame_size, fdidx, flatten=True)  # (M', n)
        #  (1, n) @ (n, M') - (1, M') -> (1, M')
        ll[k, fdidx] = frames_flat[k].dot(np.log(wji.T + 1e-13)) - wji.sum(1)[None, :]

    ll = ll.T  # (M, N)
    ll = np.clip(ll - np.max(ll, axis=0, keepdims=True), -1000.0, 1.)
    p_jk = np.exp(ll)

    membershipt_probability = p_jk / p_jk.sum(0)
    return membershipt_probability


def compute_membership_probability_frame_spase_memsaving(frames_flat,
                                                         model,
                                                         frame_size: Tuple[int, int],
                                                         drift_radius: Tuple[int, int],
                                                         frame_drifts_in_use: Optional[List[List[int]]]):
    h, w = frame_size
    pad = len(max(frame_drifts_in_use, key=len))
    frame_drifts = np.array([list(i) + [-1]*(pad-len(i)) for i in frame_drifts_in_use], dtype=np.int32)

    # (M, N)
    ll = emc_kernel.compute_log_likelihood_map_frame_sparse(frames_flat.astype(np.float32),
                                                            model.astype(np.float32),
                                                            h, w,
                                                            drift_radius[0],
                                                            drift_radius[1],
                                                            frame_drifts)

    ll = np.clip(ll - np.max(ll, axis=0, keepdims=True), -1000.0, 1.)
    p_jk = np.exp(ll)

    membershipt_probability = p_jk / p_jk.sum(0)
    return membershipt_probability


def compute_effect_merge_weights(membership_probability, drifts_in_use):
    """
    Convert membership probability to a merge-weight matrix.
    If one row of membership_probability is summed to zero (< 1e-13), then no frame contributes to the corresponding
    drift location. During the compress step, these rows should not be accounted.

    Parameters
    ----------
    membership_probability: array  (M, N)
    drifts_in_use: List[int]

    Returns
    -------
    merge_weights: array (M', N)
        M' < M if there exist rows in membership_probability that summed to zero.
    Notes
    -----

    """
    # Dynamically removing drifts where no frame contributes
    z = membership_probability.sum(1)  # (M, )
    effect_drifts_idx = np.argwhere(z > 1e-17).squeeze()
    merge_weights = membership_probability[effect_drifts_idx] / z[effect_drifts_idx][:, None]  # broadcast to (M', N)
    effect_drifts_in_use = np.array(drifts_in_use, dtype=np.uint32)[effect_drifts_idx]
    return merge_weights, effect_drifts_in_use


def merge_frames_soft(frames_flat,
                      frame_size: Tuple[int, int],
                      model_size: Tuple[int, int],
                      membership_probability: np.ndarray,
                      drift_radius: Tuple[int, int],
                      drifts_in_use: Optional[List[int]] = None):
    """
    Update patterns from frames according to the given membership_prabability.

    Parameters
    ----------
    frames_flat: 2D array in shape (N, h*w)
    frame_size: Tuple[int, int]
        the frame shape (h, w).
    model_size: Tuple[int, int]
        the model shape (H, W).
    membership_probability: array in shape (M, N)
        the membership probabilities for each frame against each drift.
    drift_radius: Tuple[int, int]
    drifts_in_use: Optional[List[int]]

    Returns
    -------
    the updated patterns in shape (M, *frame_size)
    """
    if drifts_in_use is None:
        drifts_in_use = list(range((2*drift_radius[0] + 1) * (2*drift_radius[1] + 1)))

    ec_op = ECOperator(drift_radius)

    merge_weights, effect_drifts_in_use = compute_effect_merge_weights(membership_probability, drifts_in_use)

    new_w_ji = merge_weights @ frames_flat  # (M', N) @ (N, n_pix) = (M', n_pix)
    new_expanded_model = new_w_ji.reshape(-1, *frame_size)  # (M', h, w)

    return ec_op.compress(new_expanded_model, model_size, effect_drifts_in_use)


def merge_frames_soft_memsaving(frames_flat,
                                frame_size: Tuple[int, int],
                                model_size: Tuple[int, int],
                                membership_probability: np.ndarray,
                                drift_radius: Tuple[int, int],
                                drifts_in_use: Optional[List[int]] = None):
    if not _EMC_KERNEL_INSTALLED:
        raise RuntimeError("need cpp extension to run this function")

    if drifts_in_use is None:
        drifts_in_use = list(range((2*drift_radius[0] + 1) * (2*drift_radius[1] + 1)))

    merge_weights, effect_drifts_in_use = compute_effect_merge_weights(membership_probability, drifts_in_use)
    max_drift_x, max_drift_y = drift_radius
    h, w = frame_size
    H, W = model_size

    effect_drifts_in_use = np.array(effect_drifts_in_use, dtype=np.uint32)
    model = emc_kernel.merge_frames_soft(
        frames_flat=frames_flat.astype(np.float32),
        h=h, w=w, H=H, W=W,
        drift_radius_y=max_drift_y,
        merge_weights=merge_weights.astype(np.float32),
        drifts_in_use=effect_drifts_in_use)

    return model
