"""
This module defines functions for drift-correction simulations.
i.e. these functions are usefull only we have the true model
"""
from typing import Tuple
import numpy as np
from skimage import transform as skt
import logging

from . import utils
from .core import EMC


logger = logging.getLogger("emc2d.sim")


def random_walk_trajectory(
        drift_radius: Tuple[int, int], num_steps=200, start=(0, 0), continuous=False, rand_seed=None, **kwargs):
    """
    generate translational vectors for constrained random walk (RW).
    Parameters
    ----------
    drift_radius: Tuple[int, int]
        the max value of each component of each vector
    num_steps: int
        number of steps of the RW
    start: 2-tuple or list or array
        the starting point of RW
    continuous: bool
        if True, continuous random walk is generated,
        and key word "sigma" must be given as an argument for its standard deviation.
    rand_seed: int or None, optional
        a random seed between 0 and 2**32 - 1. If none, than /dev/urandom would be used.
    Returns
    -------
    rw: 2D array in shape (num_step, 2)
    """
    np.random.seed(rand_seed)
    r = np.array(start)
    if not continuous:
        rw = np.empty(shape=(num_steps, 2), dtype=np.int)
        rw[0] = r.astype(np.int)
        for i in range(1, num_steps):
            dr = np.random.randint(low=-1, high=2, size=2)
            r = np.where(np.abs(r+dr) > np.array(drift_radius), r-dr, r+dr)
            rw[i] = r
        return rw
    else:
        rw = np.empty(shape=(num_steps, 2), dtype=np.float)
        rw[0] = r.astype(np.float)
        if "sigma" not in kwargs:
            raise ValueError("standard deviation 'sigma' must be given for continuous random walk")
        for i in range(1, num_steps):
            dr = kwargs["sigma"]*np.random.randn(2)
            while np.any(np.abs(dr) > np.array(drift_radius)):
                dr = kwargs["sigma"]*np.random.randn(2)
            r = np.where(np.abs(r+dr) > np.array(drift_radius), r-dr, r+dr)
            rw[i] = r
        return rw


def generate_frames(
        intensity, window_size, drift_radius: Tuple[int, int], num_frames: int, mean_count: float, motion_sigma: float):
    model_size = tuple(w + 2*R for w, R in zip(window_size, drift_radius))
    model = build_model(intensity, model_size, mean_count)
    # origin centered
    traj = random_walk_trajectory(drift_radius=drift_radius, num_steps=num_frames, continuous=True, sigma=motion_sigma)
    traj = np.round(traj).astype(int)
    translation_series = utils.get_translation_series(
        model, window_size=window_size, translations=traj+np.array(drift_radius))
    return np.random.poisson(translation_series), traj


def group_frames(frames, group_size: int):
    if group_size == 1:
        return frames
    num_frames = frames.shape[0]
    remainder = num_frames % group_size
    whole = int(num_frames - remainder)

    new_frames = np.array([np.sum(frames[i:i + group_size, :, :], axis=0) for i in range(0, whole, group_size)])
    return new_frames


def group_motions(traj, group_size: int, average: bool = True):
    if group_size == 1:
        return traj

    num_frames = traj.shape[0]
    remainder = num_frames % group_size
    whole = int(num_frames - remainder)

    new_traj = np.array([traj[i:i + group_size, :] for i in range(0, whole, group_size)])
    if average:
        new_traj = np.sum(new_traj, axis=1)/group_size
    return new_traj


def normalize_img_linear(img, new_min, new_max):
    """
    linearly normalizes an image such that its values are between new_min and new_max
    Parameters
    ----------
    img: numpy array
        2D image being normalized
    new_min: float
        the target minimum value
    new_max: float
        the target maximum value
    Returns
    -------
    new_img: numpy array
        the 2D normalized image
    """
    if new_max <= new_min:
        raise ValueError("new_max mast be larger than new_min")
    old_min = np.min(img)
    old_max = np.max(img)

    new_img = (img - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

    return new_img


def build_model(img, model_size, mean_count):
    """
    resizes a image and changes its mean value to build a model for simulations
    Parameters
    ----------
    img: numpy array
        the original image used as the escense of the model
    model_size: tuple or list of 2 integers
        the target model size
    mean_count: int or float
        the target mean value of the model.
    Returns
    -------
    model: numpy array
        the built model.
    """
    # resize the image by down- or up-sampling
    model = skt.resize(img, model_size, mode="constant").astype(np.float)
    model = normalize_img_linear(model, 0, 10)
    model *= mean_count / model.mean()

    return model


def mse_error(recon_traj, true_traj):
    return np.mean(np.linalg.norm(recon_traj - true_traj, axis=1)**2)


# For single run test
def run_emc_and_compute_traj_mse(frames, drift_radius, true_model, true_traj, iterations: int):
    emc = EMC(
        frames,
        frame_size=frames.shape[-2:],
        drift_radius=drift_radius,
        init_model=true_model)

    emc.run(iterations=iterations)
    emc.curr_model = true_model
    recon_traj, recon_model = emc.centre_by_reference(true_model, centre_is_origin=True)
    return mse_error(recon_traj, true_traj), recon_traj, recon_model
