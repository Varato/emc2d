import torch
import numpy as np

from collections import namedtuple

from .utils import make_drift_vectors, make_emc_translations
from .transform import make_translation_series


class MotionSpaceValueMaps:
    def __init__(self, val, xy):
        if val.shape[1] != xy.shape[1]:
            raise ValueError("val and xy must have the same second dimension")
        self.val = val  # (N, M)
        self.xy = xy    # (2, M)
        self.num_maps = val.shape[0]
        self.num_points = xy.shape[1]


def make_emc_translations(max_drift: Union[Tuple[int, int], int], coarse_graining: int = 1):
    """
    Makes an array of all per-pixel (x, y) translations within -max_drift to max_drift.

    Parameters
    ----------
    max_drift: Union[Tuple[int, int], int]
    coarse_graining: int

    Returns
    -------
    an array with shape (2, M), where M is the number of traslations being considered.
    """
    xm, ym = (max_drift, max_drift) if type(max_drift) is int else max_drift
    x_range = torch.arange(-xm, xm+1, coarse_graining)
    y_range = torch.arange(-ym, ym+1, coarse_graining)
    xx, yy = torch.meshgrid(x_range, y_range)
    return torch.stack([xx.flatten(), yy.flatten()])


def translate(img, x=0, y=0):
    xx = int(x) % img.shape[-2]
    yy = int(y) % img.shape[-1]
    ret = torch.roll(img, shifts=(xx, yy), dims=(-2, -1))
    return ret


def centre_crop(img, size):
    sx, sy = (size, size) if type(size) is int else size
    lx, ly = img.shape[-2:]
    mx, my = (lx - sx) // 2, (ly - sy) // 2
        
    ret = img[mx:mx+sx, my:my+sy]
    return ret


def make_translation_series(intensity, frame_size, trans):
    # trans (2, N) or (1, N). latter means 1D case
    if trans.ndim == 1:
        return torch.stack([
            centre_crop(
                translate(intensity, int(x), 0), size=frame_size) for x in trans.squeeze()])
    elif trans.shape[0] == 2:
        return torch.stack([
            centre_crop(
                translate(intensity, int(x), int(y)), size=frame_size) for x, y in zip(trans[0], trans[1])])
    else:
        raise ValueError


def compute_log_likelihood_maps(intensity, frames, max_drift, coarse_graining):
    # frames: (N, h, w)
    # translations: (2, M)
    N = frames.shape[0]
    M = translations.shape[1]
    frame_size = frames.shape[-2:]
    translations = make_emc_translations(max_drift, coarse_graining)
    trans_series = make_translation_series(intensity, frame_size, translations) # (M, h, w)
    
    w_ji = trans_series.reshape(M, -1)  # (M, n_pix)
    y_ji = frames.reshape(N, -1)        # (N, n_pix)
    
    w_j = w_ji.sum(-1, keepdims=True)  #(M, 1)
    log_w_ji = np.log(w_ji + 1e-17)    #(M, n_pix)
    
    ll = log_w_ji @ y_ji.T - w_j         # (M, N)
    ll = torch.clamp(ll - torch.max(ll, dim=0, keepdim=True)[0], -100.0, 1.).T  # (N, M)
    
    return MotionSpaceValueMaps(ll, translations)


def find_max_log_likelihood_traj(log_likelihood_maps: MotionSpaceValueMaps):
    """
    log_likelihood_maps.val: (N, M)
    log_likelihood_maps.xy: (2, M)
    """
    idx = torch.argmax(log_likelihood_maps.val, dim=1)
    traj = log_likelihood_maps.xy[:, idx]
    return traj  # (2, N)


class EMC(object):

    def __init__(self, frames, max_drift, init_model='sum'):
        self.frames = tf.constant(frames, dtype=tf.uint16)

        self.n_frames = frames.shape[0]
        self.frame_size = frames.shape[1:]
        self.max_drift = max_drift
        self.curr_model = self.initialize_model(init_model)
        self.model_size = self.curr_model.shape

        self.drifts = make_drift_vectors(max_drift, origin='corner')
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

        expected_model_size = (self.frame_size[0] + 2*self.max_drift + 1,
                               self.frame_size[1] + 2*self.max_drift + 1)

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

    def expand(self, model, drift_indices):
        num_drifts = len(drift_indices)
        window_size = self.frame_size
        expanded_model = np.empty(shape=(num_drifts, *window_size))
        for i in drift_indices:
            s = self.drifts[i]
            expanded_model[i, :, :] = model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]]
        return expanded_model

    def compress(self, expanded_model, drift_indices):
        window_size = self.frame_size

        model = np.zeros(shape=self.model_size)
        weights = np.zeros_like(model)
        for k, i in enumerate(drift_indices):
            s = self.drifts[i]
            model[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += expanded_model[k]
            weights[s[0]:s[0]+window_size[0], s[1]:s[1]+window_size[1]] += 1

        return model / np.where(weights == 0., 1.0, weights)

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





if __name__ == "__main__":
    pass
