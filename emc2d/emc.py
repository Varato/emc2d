from typing import Tuple, Union, List
import numpy as np
import time

from .utils import make_drift_vectors


class EMC(object):
    def __init__(self, frames: np.ndarray, max_drift: int, init_model: Union[str, np.ndarray] = 'sum'):
        """
        Parameters
        ----------
        frames: ndarray in shape (n_frames, h, w)
        max_drift: int
        init_model: str or ndarray
            If it's a string, it should be either 'sum' or 'random'
        """
        self.frames = frames
        self.n_frames = frames.shape[0]
        self.frame_size = frames.shape[1:]
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
        self._mask = None

    def run(self, iterations: int, memsaving: bool = False, verbose=True):
        history = {'model_power': []}
        for i in range(iterations):
            start = time.time()
            self.one_step(memsaving)
            end = time.time()
            power = self.curr_model.mean()
            if verbose:
                print(f"iter {i+1} / {iterations}: model_power = {power:.3f}, time used = {end-start:.3f} s")
            history['model_power'].append(self.curr_model.mean())
        return history

    def one_step(self, memsaving: bool = False):
        membership_prabability = self.expand_memsaving() if memsaving else self.expand()
        patterns = self.maximize(membership_prabability)
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
            model = self.frames.sum(0)
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
        expanded_model = self._expand()
        n_drifts = expanded_model.shape[0]
        x_ki = self.frames.reshape(self.n_frames, -1)  # (n_frames, n_pix)
        w_ji = expanded_model.reshape(n_drifts, -1)    # (n_drifts, n_pix)
        w_j = w_ji.sum(1, keepdims=True)               # (n_drifts, 1)
        log_wji = np.log(w_ji + 1e-17)

        LL = np.matmul(log_wji, x_ki.T) - w_j         # (n_drifts, n_frames)
        LL = np.clip(LL - np.max(LL, axis=0, keepdims=True), -100.0, 1.)
        p_jk = np.exp(LL)

        membershipt_probability = p_jk / p_jk.sum(0)
        return membershipt_probability

    def expand_memsaving(self) -> np.ndarray:
        """
        Expands current model into patterns, and compute the membership probabilities for each frame.

        It differs from `expand` in the following way: rather than store the full set of patterns, it
        computes the membership probabilities on the fly. In this Python implementation, this saves memory
        but may be time-inefficient.

        Parameters
        ----------
        
        Returns
        -------
        membership probabilities as a 2D array in shape (n_drifts, n_frames).
        """
        n_drifts = len(self.drifts_in_use)
        window_size = self.frame_size
        membership_probability = np.empty(shape=(n_drifts, self.n_frames), dtype=np.float)
        x_ki = self.frames.reshape(self.n_frames, -1) # (n_frames, n_pix)

        for j, idx in enumerate(self.drifts_in_use):
            s = self.drifts[idx]
            pattern     = self.curr_model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]].reshape(-1,)
            log_pattern = np.log(pattern + 1e-17)
            LL = np.matmul(log_pattern, x_ki.T) - np.sum(pattern)  # (n_frames,)
            LL = np.clip(LL - np.max(LL), -100.0, 1.)
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
        x_ki = self.frames.reshape(self.n_frames, -1)                       # (n_frames, n_pix)

        new_w_ji = np.matmul(weights_jk, x_ki)  # (n_drifts, n_pix)
        new_expanded_model = new_w_ji.reshape(n_drifts, *self.frame_size)

        return new_expanded_model

    def _expand(self) -> np.ndarray:
        n_drifts = len(self.drifts_in_use)

        window_size = self.frame_size
        expanded_model = np.empty(shape=(n_drifts, *window_size), dtype=np.float)
        for j, i in enumerate(self.drifts_in_use):
            s = self.drifts[i]
            expanded_model[j] = self.curr_model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]]

        return expanded_model





