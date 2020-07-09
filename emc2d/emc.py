import numpy as np
from numpy import ma

from .utils import make_drift_vectors
from .transform import Drift


class EMC(object):

    def __init__(self, frames, max_drift, init_model='sum'):
        self.frames = frames
        self.n_frames = frames.shape[0]
        self.frame_size = frames.shape[1:]
        self.max_drift = max_drift
        self.curr_model = self.initialize_model(init_model)
        self.model_size = self.curr_model.shape

        self.drifts = make_drift_vectors(max_drift, origin='corner')
        self.transforms = Drift(self.model_size, self.frame_size)
        self._mask = None

    def initialize_model(self, init_model):
        """
        regularise the initial model, including pad the initial model according to img_size and max_drift.


        Parameters
        ----------
        init_model: str or numpy array
        Returns
        -------
        the regularized initial model
        """
        # model_size - frame_size = 2*max_drift
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

        init_shape = model.shape

        assert model is not None

        # if any dimension of the given model is smaller than the pexpected shape, pad that dimension.
        is_smaller = [l < lt for l, lt in zip(init_shape, expected_model_size)]
        if any(is_smaller):
            px = expected_model_size[0] - init_shape[0] if is_smaller[0] else 0
            py = expected_model_size[1] - init_shape[1] if is_smaller[1] else 0
            pad_width = (
                (px//2, px//2) if px%2 == 0 else (px//2 + 1, px//2), 
                (py//2, py//2) if py%2 == 0 else (py//2 + 1, py//2))
            return np.pad(model, pad_width, mode='constant', constant_values=0)
        # if both dimensions of the given model is larger than or equal to the target size, crop it.
        else:
            margin = [init_shape[i] - expected_model_size[i] for i in range(2)]
            start_x = margin[0]//2 if margin[0]%2 == 0 else margin[0]//2 + 1
            start_y = margin[1]//2 if margin[1]%2 == 0 else margin[1]//2 + 1
            return model[start_x:start_x+expected_model_size[0], start_y:start_y+expected_model_size[1]]

    def expand(self, model, drift_indices=None):
        # return self.transforms.forward(model, self.drifts)
        if drift_indices is None:
            drift_indices = np.arange(0, len(self.drifts))

        num_drifts = len(drift_indices)
        window_size = self.frame_size
        expanded_model = np.zeros(shape=(num_drifts, *window_size), dtype=np.float)
        for k, i in enumerate(drift_indices):
            s = self.drifts[i]
            expanded_model[k] = model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]]
        return expanded_model

    def compress(self, expanded_model, drift_indices=None):
        # return self.transforms.backward(expanded_model, self.drifts)
        if drift_indices is None:
            drift_indices = np.arange(0, len(self.drifts))

        window_size = self.frame_size
        model   = np.zeros(shape=self.model_size, dtype=np.float)
        weights = np.zeros(shape=self.model_size, dtype=np.float)
        for k, i in enumerate(drift_indices):
            s = self.drifts[i]
            model  [s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += expanded_model[k]
            weights[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += 1.0

        return model / np.where(weights>0., weights, 1.)

    def maximize(self, expanded_model):
        npix = self.frame_size[0] * self.frame_size[1]
        w_ji = expanded_model.reshape(-1, npix)  # (num_transforms, n*m)
        if self._mask is not None:
            w_j = np.sum(w_ji[:, self._mask], axis=1, keepdims=True)
        else:
            w_j = np.sum(w_ji, axis=1, keepdims=True)  # (num_transforms, 1)

        logw_ji = ma.log(w_ji).filled(-39)  # (num_transforms, n*m)

        x_ki = self.frame_size.reshape(-1, npix)
        # (num_transforms, num_frames) = (num_transforms, n*m) @ (n*m, num_frames) - (num_transforms, 1)
        p_jk = logw_ji @ x_ki.T - w_j
        p_jk = np.clip(p_jk - np.max(p_jk, axis=0, keepdims=True), -100.0, 1.)
        p_jk = np.exp(p_jk)
        p_jk /= np.sum(p_jk, axis=0, keepdims=True)

        weights_jk = p_jk / p_jk.sum(1, keepdims=True)

        new_w_ji = weights_jk @ x_ki  # (num_transforms, n*m)
        new_expanded_model = new_w_ji.reshape(*expanded_model.shape)

        return new_expanded_model, p_jk






