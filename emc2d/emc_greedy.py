import time
import numpy as np
from typing import Tuple, Union, List
from scipy.sparse import csr_matrix
from scipy.special import softmax

import logging

from .misc import *


class EMCGreedy(object):
    def __init__(self, frames: np.ndarray, frame_size: Tuple[int, int], max_drift: int, init_model: Union[str, np.ndarray] = 'sum'):
        """
        Parameters
        ----------
        frames: ndarray or csr_matrix in shape (num_frames, h, w) or (num_frames, h*w)
        max_drift: int
        init_model: str or ndarray
            If it's a string, it should be either 'sum' or 'random'
        """
        self.num_frames = frames.shape[0]
        self.frames = frames.reshape(self.num_frames, -1)  # (n_frames, n_pix)
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
        # self.membership_prabability = None

        # to hold the estimated traj
        self.curr_traj = None

    def run(self, iterations: int, verbose=True):
        history = {'model_mean': [], 'convergence':[]}
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


    def one_step(self, searching_radias: int):
        self.curr_traj = self.find_trajectory(searching_radias)
        self.curr_model = self.merge_frames(self.curr_traj)


    def estimate_first_frame_position(self):
        dim_x, dim_y = (2*self.max_drift + 1, ) * 2

        first_frame = self.frames[0][:, None]  #(n_pix, 1)
        expanded_model = self.ec_op.expand(self.curr_model, self.frame_size, self.drifts_in_use, flatten=True)
        
        ll = np.log(expanded_model + 1e-13) @ first_frame - expanded_model.sum(1, keepdims=True)
        pos_dist = softmax(ll.flatten())
        pos_idx = np.argmax(pos_dist)
        
        x, y = pos_idx // dim_y, pos_idx % dim_y
        return x, y

    def find_trajectory(self, searching_radias: int = 5):
        dim_x, dim_y = (2*self.max_drift + 1, ) * 2
        x, y = self.estimate_first_frame_position()

        traj = np.empty(shape=(self.num_frames, 2), dtype=np.int)
        traj[0] = [x, y]
        for i in range(1, self.num_frames):
            nbs_x, nbs_y = find_neighbours(dim_x, dim_y, x, y, searching_radias)
            nbs_idx = (nbs_x * dim_y + nbs_y).flatten()
            sparse_expanded_model = self.ec_op.expand(self.curr_model, self.frame_size, nbs_idx, flatten=True)
            ll = np.log(sparse_expanded_model + 1e-13) @ self.frames[i][:, None] - sparse_expanded_model.sum(1, keepdims=True)
            pos_dist = softmax(ll.flatten())
            pos_idx = nbs_idx[np.argmax(pos_dist)]
            x, y = pos_idx // dim_y, pos_idx % dim_y
            traj[i] = [x, y]
        return traj

    def merge_frames(self, trajectory):
        frames = self.frames.reshape(self.num_frames, *self.frame_size)

        model   = np.zeros(shape=self.model_size, dtype=np.float)
        weights = np.zeros(shape=self.model_size, dtype=np.float)
        for i in range(self.num_frames):
            s = trajectory[i]
            model  [s[0]:s[0]+self.frame_size[0], s[1]:s[1]+self.frame_size[1]] += frames[i]
            weights[s[0]:s[0]+self.frame_size[0], s[1]:s[1]+self.frame_size[1]] += 1.0
        return model / np.where(weights>0., weights, 1.)


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
        return model_reshape(model, expected_model_size)

    def maximum_likelihood_drifts(self):
        if self.membership_prabability is None:
            raise RuntimeError("EMC must be run before estimating drifts")
        return maximum_likelihood_drifts(self.membership_prabability, self.ec_op.all_drifts, self.drifts_in_use)

    def centre_by_first_frame(self):
        frame_positions = self.maximum_likelihood_drifts()
        calibrating_drift, recon_drifts = centre_by_first_frame(frame_positions, self.max_drift, centre_is_origin=True)
        return recon_drifts, np.roll(self.curr_model, shift=calibrating_drift, axis=(-2, -1))

    def centre_by_reference(self, reference, centre_is_origin=True):
        calibrating_drift, recon_drifts = centre_by_reference(
            frame_positions, self.max_drift, 
            self.frame_size, self.curr_model, reference, self.drifts_in_use, centre_is_origin)
        return recon_drifts, np.roll(self.curr_model, shift=calibrating_drift, axis=(-2, -1))


def find_neighbours(dim_x, dim_y, x, y, n=1):
    if n <= x <= dim_x - 1 - n:
        xi = range(-n, n+1)
    elif x >= n:
        xi = range(-n, dim_x-x)
    else:
        xi = range(-x, n+1)
        
    if n <= y <= dim_y - 1 - n:
        yi = range(-n, n+1)
    elif y >= n:
        yi = range(-n, dim_y-y)
    else:
        yi = range(-y, n+1)
        
    nb_idx = np.array([[(x + a, y + b) for b in yi] for a in xi])
    return nb_idx[..., 0], nb_idx[..., 1]