from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from . import utils
from . import core


def show_coarse_grained_drift_space(drift_radius: Tuple[int, int], multiple: Tuple[int, int], ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
        
    coarsed_drift_indices = utils.drift_space_coarse_grain(drift_radius, multiple)
    cx, cy = utils.drift_indices_to_locations(drift_radius, coarsed_drift_indices)
        
    ax.set_aspect('equal')
    h, w = [2*r + 1 for r in drift_radius]
    mat = np.zeros(shape=(h, w), dtype=np.int)
    mat[cx, cy] = 1
    ax.imshow(mat, cmap='gray', vmin=0, vmax=1)
    ax.set_ylabel('x')
    ax.set_xlabel('y')
    return fig
    

def show_membership_probability(membership_probability,
                                frame_index: int,
                                drift_radius: Tuple[int, int],
                                drifts_in_use: Optional[List[int]] = None,
                                ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.set_ylabel('x')
    ax.set_xlabel('y')
    pmat = utils.fold_likelihood_map(membership_probability, drift_radius, drifts_in_use)
    frame_mat = pmat[frame_index]
    print("??")
    ax.imshow(frame_mat, vmin=0, vmax=0.01)
    return fig, pmat


def show_emc_state(emc: core.EMC):
    recon_cropped = utils.centre_crop(emc.curr_model, emc.frame_size)
    frames = np.array(emc.frames.todense()) if type(emc.frames) is csr_matrix else np.array(emc.frames)
    frames_sum = frames.reshape(emc.num_frames, *emc.frame_size).sum(0)

    fig, axes = plt.subplots(ncols=5, figsize=(10, 3))
    axes[0].plot(emc.history['model_mean'])
    axes[1].plot(emc.history['convergence'])
    axes[2].hist(emc.curr_model.reshape(-1,), bins=50)
    axes[3].imshow(frames_sum)
    axes[4].imshow(recon_cropped)

    axes[0].set_title('model mean')
    axes[1].set_title('convergence')
    axes[2].set_title("recon value hist")
    axes[3].set_title('sum')
    axes[4].set_title('recon')
    return fig


def show_maximum_likelihood_drifts(emc: core.EMC, reference=None, true_traj=None, axes=None):
    if reference is not None:
        recon_shifts, recon_model = emc.centre_by_reference(reference)
    else:
        recon_shifts, recon_model = emc.centre_by_first_frame()

    fig = None
    if axes is None:
        fig, axes = plt.subplots(nrows=2, sharex='all')

    axes[0].plot(recon_shifts[:, 0], 'r', label='max lik x')
    axes[1].plot(recon_shifts[:, 1], 'r', label='max lik y')
    if true_traj is not None:
        axes[0].plot(true_traj[:, 0], 'b--', alpha=0.5, label='true x')
        axes[1].plot(true_traj[:, 1], 'b--', alpha=0.5, label='true y')
    return fig
