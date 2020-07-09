"""
This module defines functions for drift-correction simulations.
i.e. these functions are usefull only we have the true model
"""
import numpy as np
from skimage import transform as skt
import logging

from emc2d.transform import Drift
from emc2d.utils import make_drift_vectors

logger = logging.getLogger("emlab.processing.emc_motioncorr.simtools")


def frames_with_uniform_drifts(model, img_size, num_frames, max_drift, mean_count, rand_seed=None, add_background=False):
    """
    generated uniform-randomly drifted data from given model.
    Parameters
    ----------
    model: numpy array
        the 2D model used to generate data.
    img_size: tuple or list
        the 2D image size of each generated data entry
    num_frames: int
        number of images being generated.
    max_drift: int
        maximum drift in number of pixels
    mean_count: float
        the mean value of each data entry
    rand_seed: int or None, optional
        a random seed between 0 and 2**32 - 1. If none, than /dev/urandom would be used.
    add_background: bool
        If true, a white nose is added on the model before poissonian sampling
    Returns
    -------
    data: numpy array
        The first dimension corresponds to different data entries.
    shuffled_drift_table: numpy array
        The corresponding drift vectors for each data entry.
    """
    np.random.seed(rand_seed)
    drift_table = make_drift_vectors(max_drift)
    num_drifts = drift_table.shape[0]

    rand_indx = np.random.randint(0, num_drifts, size=num_frames)
    shuffled_drift_table = drift_table[rand_indx]
    drift = Drift(model.shape, img_size)
    data = drift.forward(model, shuffled_drift_table)
    data[data <= 1e-7] = 1e-7
    if add_background:
        data += 2*np.random.rand(*data.shape)

    data *= mean_count / data.mean()
    data = np.random.poisson(data)

    return data, shuffled_drift_table


def random_walk_trajectory(max_drift=10, num_steps=200, start=(0, 0), continuous=False, rand_seed=None, **kwargs):
    """
    generate translational vectors for constrained random walk (RW).
    Parameters
    ----------
    max_drift: int
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
            r = np.where(np.abs(r+dr) > max_drift, r-dr, r+dr)
            rw[i] = r
        return rw
    else:
        rw = np.empty(shape=(num_steps, 2), dtype=np.float)
        rw[0] = r.astype(np.float)
        if "sigma" not in kwargs:
            raise ValueError("standard deviation 'sigma' must be given for continuous random walk")
        for i in range(1, num_steps):
            dr = kwargs["sigma"]*np.random.randn(2)
            r = np.where(np.abs(r+dr) > max_drift, r-dr, r+dr)
            rw[i] = r
        return rw


def frames_with_random_walk_drifts(model, img_size, random_walk_trj, mean_count, add_background=False, rand_seed=None):
    """
    Parameters
    ----------
    model: numpy array
        the 2D true model
    img_size: 2-tuple or list
        the target image size
    random_walk_trj: numpy array
        the random walk vectors
    mean_count: int or float
        the mean value of each pixel of each generated image
    add_background: bool
        If true, a white nose is added on the model before poissonian sampling
    rand_seed: int or None, optional
        a random seed between 0 and 2**32 - 1. If none, than /dev/urandom would be used.
    Returns
    -------
        data: 3D array (num_step, n, m)
    """
    np.random.seed(rand_seed)
    import time
    tic = time.time()

    num_data = random_walk_trj.shape[0]
    transform = Drift(model.shape, img_size)
    # accept continuous random walk, but round off it.
    if random_walk_trj.dtype == np.float:
        random_walk_trj = np.round(random_walk_trj).astype(np.int)

    data = transform.forward(model, random_walk_trj)
    data[data < 1e-7] = 1e-7
    if add_background:
        data += 2*np.random.rand(*data.shape)
    data *= mean_count / data.mean()
    data = np.random.poisson(data).astype(np.int)
    toc = time.time()

    logger.info("{} images generated, time used = {:.3f}s".format(num_data, toc - tic))

    return data


def group_frames(frames, frame_drifts, group_size, average_drifts=False):
    """
    groups frames into subgroups and sum them, so that blurred frames are generated.
    Parameters
    ----------
    frames: numpy array
        the frames being grouped
    frame_drifts: numpy array
        the drift vectors that corresponds to each frame
    group_size: int
        specifies how many consecutive frames are to be grouped together
    average_drifts: bool
        If true, the drifts corresponding to each frame group will be averaged.
    Returns
    -------
    new_frames: numpy array
        the grouped (blurred) frames
    grouped_drifts: numpy array
    """
    if group_size == 1:
        return frames, frame_drifts
    num_data = frames.shape[0]
    remainder = num_data % group_size
    whole = int(num_data - remainder)

    new_frames = [np.sum(frames[i:i + group_size, :, :], axis=0) for i in range(0, whole, group_size)]
    grouped_drifts = [frame_drifts[i:i + group_size, :] for i in range(0, whole, group_size)]
    # if remainder != 0:
    #     new_frames.append(np.sum(frames[whole:, :, :], axis=0))
    #     grouped_drifts.append(np.sum(frame_drifts[whole:, :], axis=0))
    new_frames = np.array(new_frames, dtype=np.int)
    grouped_drifts = np.array(grouped_drifts)
    if average_drifts:
        grouped_drifts = np.sum(grouped_drifts, axis=1)/group_size
    return new_frames, grouped_drifts


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