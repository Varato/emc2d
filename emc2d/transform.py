from typing import Tuple, List
import numpy as np
# from scipy.ndimage import rotate

from .utils import make_drift_vectors


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
            expanded_model[j] = model[s[0]:s[0] + window_size[0], s[1]:s[1] + window_size[1]]

        if flatten:
            return expanded_model.reshape(n_drifts, -1)
        else:
            return expanded_model

    def compress(self, expanded_model, model_size: Tuple[int, int], drifts_in_use: List[int] = None):
        """
        Compresses an expaned_model (patterns) into a model according to the corresponding drifts.

        Parameters
        ----------
        expanded_model: array in shape (M, h, w)
        model_size: Tuple[int, int]
        drifts_in_use: List[int]

        Returns
        -------
        an assembled model as 2D array
        """
        if drifts_in_use is None:
            drifts_in_use = list(range(self.num_all_drifts))

        window_size = expanded_model.shape[-2:]
        model = np.zeros(shape=model_size, dtype=np.float)
        weights = np.zeros(shape=model_size, dtype=np.float)
        for k, i in enumerate(drifts_in_use):
            s = self.all_drifts[i]
            model[s[0]:s[0] + window_size[0], s[1]:s[1] + window_size[1]] += expanded_model[k]
            weights[s[0]:s[0] + window_size[0], s[1]:s[1] + window_size[1]] += 1.0
        return model / np.where(weights > 0., weights, 1.)


class Drift(object):
    """
    This class defines the drift transformation for EMC.
    """
    def __init__(self, model_size, img_size):
        self._model_size = model_size
        self._img_size = img_size

    @staticmethod
    def forward_single(model, drift, x_min, y_min, img_size):
        x, y = -np.array(drift).astype(np.int)
        view = model[-x_min+x:-x_min+x+img_size[0],
                     -y_min+y:-y_min+y+img_size[1]]
        return view

    def forward(self, model, drifts):
        """
        expands a model into a set of drifted images with shape img_size.
        Parameters
        ----------
        model: numpy array
            2D model for EMC
        drifts: numpy array
            specifies drift vector(s). If ndim=1, then only 1 drift vector is interpreted.
            If ndim=2, the first dimension corresponds to different vector(s).
        Returns
        -------
        views: numpy array
            drifted image(s). If more than one drifts are given, the first dimension of views
            corresponds to different drifts.
        """

        mm, nn = model.shape
        if not (mm > self._img_size[0] and nn > self._img_size[1]):
            raise ValueError("model size must be larger than image size")

        x_min, _ = self.find_max_drift(mm, self._img_size[0])
        y_min, _ = self.find_max_drift(nn, self._img_size[1])

        drifts = np.array(drifts).astype(np.int)
        if drifts.ndim == 1 and drifts.shape[0] == 2:
            return self.forward_single(model, drifts, x_min, y_min, self._img_size)

        elif drifts.ndim == 2 and drifts.shape[1] == 2:
            num_drifts = drifts.shape[0]
            views = np.empty(shape=(num_drifts, *self._img_size))
            for i, d in enumerate(drifts):
                views[i] = self.forward_single(model, d, x_min, y_min, self._img_size)
            return views

        else:
            raise ValueError("drifts must be array in shape (num_drifts, 2)")

    def backward(self, images, drifts):
        """
        compress different views (images) into a canvas with size model_size.
        Parameters
        ----------
        images: numpy array
            drifted images.
        drifts: numpy array
            specifies drift vector(s) for each image.
            If ndim = 1, then there must be only one image.
            If ndim = 2, the first dimension corresponds to different images.
        Returns
        -------
        canvas: numpy array
            the compressed model with size as model_size
        """
        m, n = images.shape[-2:]

        x_min, _ = self.find_max_drift(self._model_size[0], m)
        y_min, _ = self.find_max_drift(self._model_size[1], n)

        canvas = np.zeros(shape=self._model_size)
        weights = np.zeros_like(canvas)
        if drifts.ndim == 2 and images.ndim == 3:
            for i, v in enumerate(drifts):
                x, y = -v.astype(np.int)
                canvas[-x_min+x:-x_min+x+n, -y_min+y:-y_min+y+m] += images[i]
                weights[-x_min+x:-x_min+x+n, -y_min+y:-y_min+y+m] += 1.0
        else:
            raise ValueError("shapes of views and drifts cannot match")

        return canvas / np.where(weights > 0, weights, 1.0)

    @staticmethod
    def find_max_drift(n, window_size):
        """
        Given a 1D window with size window_size and a 1D array with size n,
        this function gives the range that the window can be drifted on the n-array.
        Parameters
        ----------
        n: int
            array size n
        window_size: int
            array size, N >n
        Returns
        -------
        x_min: int
            minimum translation vector
        x_max: int
            maximum translation vector
        Notes
        -----
        when translation = 0, the n-window is at centre of the N-model.
        Centre is defined as follows:
            if N-n is even, then the leftover pixels at two sides of the window are of same number.
            if N-n is odd, then left side has one more pixel than right side.
        """

        x_max = (n - window_size) // 2
        x_min = -x_max if (n - window_size) % 2 == 0 else -x_max - 1

        return x_min, x_max
