import time
import numpy as np
from typing import Tuple, Union, List
from scipy.sparse import csr_matrix

import logging

from .utils import make_drift_vectors

logger = logging.getLogger('emc2d')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class EMC(object):
    def __init__(self, frames: Union[np.ndarray, csr_matrix], frame_size: Tuple[int, int], max_drift: int, init_model: Union[str, np.ndarray] = 'sum'):
        """
        Parameters
        ----------
        frames: ndarray or csr_matrix in shape (n_frames, h, w) or (n_frames, h*w)
        max_drift: int
        init_model: str or ndarray
            If it's a string, it should be either 'sum' or 'random'
        """
        self.frames = EMC.vectorize_data(frames)  # (n_frames, n_pix)
        self.n_frames = frames.shape[0]
        self.frame_size = frame_size
        self.max_drift = max_drift

        # model_size - frame_size = 2*max_drift
        self.model_size = (self.frame_size[0] + 2*self.max_drift,
                           self.frame_size[1] + 2*self.max_drift)

        # initialize model and assure its size is correct
        model = self.initialize_model(init_model)
        self.curr_model = EMC.model_reshape(model, self.model_size)

        self.drifts = make_drift_vectors(max_drift, origin='corner')
        self.n_drifts_total = self.drifts.shape[0]
        # this property is used to select a subset of drifts to be taken into account
        self.drifts_in_use: List[int] = list(range(0, self.n_drifts_total))
        self.membership_prabability = None
        self._mask = None

    def run(self, iterations: int, memsaving: bool = False, verbose=True):
        history = {'model_mean': [], 'convergence':[]}
        for i in range(iterations):
            last_model = self.curr_model
            start = time.time()
            self.one_step(memsaving)
            end = time.time()
            power = self.curr_model.mean()
            convergence = np.mean((last_model - self.curr_model)**2)
            if verbose:
                logger.info(f"iter {i+1} / {iterations}: model mean = {power:.3f}, time used = {end-start:.3f} s")
            history['model_mean'].append(power)
            history['convergence'].append(convergence)
        return history

    def one_step(self, memsaving: bool = False):
        self.membership_prabability = self.expand_memsaving() if memsaving else self.expand()
        patterns = self.maximize(self.membership_prabability)
        self.curr_model = self.compress(patterns)

    def update_model(self, model):
        if model.shape != self.model_size:
            model = EMC.model_reshape(model, self.model_size)
        self.curr_model = model

    def using_drifts(self, drift_indices: List[int]):
        self.drifts_in_use = drift_indices

    @staticmethod
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

        model = None
        if (type(init_model) is str) and init_model == 'sum':
            model = self.frames.sum(0).reshape(*self.frame_size)
        elif type(init_model) is np.ndarray:
            if not init_model.ndim == 2:
                raise ValueError("initial_model has to be a 2D array.")
            model = init_model
        else:
            raise ValueError("unknown initial model type. initial model can be 'random', 'sum', or a numpy array.")

        assert model is not None
        return EMC.model_reshape(model, expected_model_size)

    def expand(self) -> np.ndarray:
        """
        Expands current model into patterns, and compute the membership probabilities for each frame.

        Parameters
        ----------

        Returns
        -------
        membership probabilities as a 2D array in shape (n_drifts, n_frames).
        """
        expanded_model = self._expand(self.curr_model)
        n_drifts = expanded_model.shape[0]
        w_ji = expanded_model.reshape(n_drifts, -1)    # (n_drifts, n_pix)
        w_j = w_ji.sum(1, keepdims=True)               # (n_drifts, 1)
        log_wji = np.log(w_ji + 1e-17)

        LL = self.frames.dot(log_wji.T).T - w_j         # (n_drifts, n_frames)
        LL = np.clip(LL - np.max(LL, axis=0, keepdims=True), -600.0, 1.)
        p_jk = np.exp(LL)

        membershipt_probability = p_jk / p_jk.sum(0)
        return membershipt_probability

    def expand_memsaving(self) -> np.ndarray:
        """
        Expands current model into patterns, and compute the membership probabilities for each frame.

        It differs from `expand` in the following way: rather than store the full set of patterns, it
        computes the membership probabilities on the fly. In this Python implementation, this saves memory
        but is time-inefficient.

        Parameters
        ----------
        
        Returns
        -------
        membership probabilities as a 2D array in shape (n_drifts, n_frames).
        """
        n_drifts = len(self.drifts_in_use)
        window_size = self.frame_size
        membership_probability = np.empty(shape=(n_drifts, self.n_frames), dtype=np.float)

        for j, idx in enumerate(self.drifts_in_use):
            s = self.drifts[idx]
            pattern     = self.curr_model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]].reshape(-1,)
            log_pattern = np.log(pattern + 1e-17)
            LL = log_pattern @ self.frames.T - np.sum(pattern)  # (n_frames,)
            LL = np.clip(LL - np.max(LL), -600.0, 1.)
            membership_probability[j, :] = np.exp(LL)

        membership_probability /= membership_probability.sum(0, keepdims=True)

        return membership_probability

    def compress(self, expanded_model: np.ndarray) -> np.ndarray:
        """
        Compresses an expaned_model (patterns) into a model according to the corresponding drifts.

        Parameters
        ----------
        expanded_model: 3D array in shape (n_drifts, h, w)
    
        Returns
        -------
        an assembled model as 2D array
        """

        window_size = self.frame_size
        model   = np.zeros(shape=self.model_size, dtype=np.float)
        weights = np.zeros(shape=self.model_size, dtype=np.float)
        for k, i in enumerate(self.drifts_in_use):
            s = self.drifts[i]
            model  [s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += expanded_model[k]
            weights[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += 1.0

        return model / np.where(weights>0., weights, 1.)

    def maximize(self, membership_probability: np.ndarray):
        """
        Update patterns from frames according to the given membership_prabability.

        Parameters
        ----------
        membership_probability: 2D array in shape (n_drifts, n_frames)
            the membership probabilities for each frame against each drift.

        Returns
        -------
        the updated patterns in shape (n_drifts, *frame_size)
        """

        n_drifts = membership_probability.shape[0]
        weights_jk = membership_probability / membership_probability.sum(1, keepdims=True) # (n_drifts, n_frames)

        new_w_ji = weights_jk @ self.frames  # (n_drifts, n_frames) @ (n_frames, n_pix) = (n_drifts, n_pix)
        new_expanded_model = new_w_ji.reshape(n_drifts, *self.frame_size)

        return new_expanded_model

    def _expand(self, model) -> np.ndarray:
        n_drifts = len(self.drifts_in_use)

        window_size = self.frame_size
        expanded_model = np.empty(shape=(n_drifts, *window_size), dtype=np.float)
        for j, i in enumerate(self.drifts_in_use):
            s = self.drifts[i]
            expanded_model[j] = model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]]

        return expanded_model

    @staticmethod
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

    def maximum_likelihood_drifts(self):
        if self.membership_prabability is None:
            raise RuntimeError("EMC has to be run before evaluating drifts")
        
        frame_position_indices = np.argmax(self.membership_prabability, axis=0)
        frame_positions = np.array([self.drifts[self.drifts_in_use[i]] for i in frame_position_indices])
        return frame_positions

    def calibrate_drifts_with_reference(self, reference, centre_is_origin=True):
        num_drifts = len(self.drifts_in_use)
        expanded_model = self._expand(self.curr_model).reshape(num_drifts, -1)
        expanded_ref = self._expand(reference).reshape(num_drifts, -1)

        centre_drift_index = self.max_drift + self.max_drift * (2*self.max_drift + 1)
        if centre_drift_index not in self.drifts_in_use:
            raise RuntimeError("centre drift is not within the EMC's view. Check the EMC.drift_in_use property")
        idx = self.drifts_in_use.index(centre_drift_index)
        centre_ref = expanded_ref[idx] # (N,)

        n1 = np.linalg.norm(centre_ref)
        n2 = np.linalg.norm(expanded_model, axis=1, keepdims=True)  #(M, 1)

        v1 = centre_ref / n1         #(N,)
        v2 = expanded_model / n2     #(M, N)

        diff = np.linalg.norm(v1[None, :] - v2, axis=1) #(M, )
        recon_drift_centre_index = np.argmin(diff)
        recon_centre_drift = self.drifts[self.drifts_in_use[recon_drift_centre_index]]

        calibrating_shift = np.array([self.max_drift, self.max_drift]) - recon_centre_drift 
        return calibrating_shift

    def centre_by_first_frame(self, centre_is_origin=True):
        frame_positions = self.maximum_likelihood_drifts()
        first_frame_position = frame_positions[0]
        calibrating_shift = np.array([self.max_drift, self.max_drift]) - first_frame_position
        frame_positions += calibrating_shift
        if centre_is_origin:
            frame_positions -= self.max_drift
        return frame_positions, np.roll(self.curr_model, shift=calibrating_shift, axis=(-2, -1))

    def centre_by_reference(self, reference, centre_is_origin=True):
        frame_positions = self.maximum_likelihood_drifts()

        calibrating_shift = self.calibrate_drifts_with_reference(reference)
        frame_positions += calibrating_shift
        if centre_is_origin:
            frame_positions -= self.max_drift
        return frame_positions, np.roll(self.curr_model, shift=calibrating_shift, axis=(-2, -1))








