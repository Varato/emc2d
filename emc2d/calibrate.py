from typing import List, Tuple, Union
import logging
import numpy as np
from scipy.sparse import csr_matrix

from .transform import ECOperator

logger = logging.getLogger('emc2d.misc')


def maximum_likelihood_drifts(membership_probability, all_drifts, drifts_in_use: List[int]):
    """
    Parameters
    ----------
    membership_probability: array
        The membership probability matrix computed in expectation maximization
        shape: (M, N), where M is the number of all posible positions of drifts and N is the number of frames.
    all_drifts: array
        All drift positions within the drift space
        shape: (M, 2)
    drifts_in_use: List[int]
        A list of indices for all_drifts, specify the drifts in use.

    Returns
    -------
    frame_positions: array
        shape: (N, 2)
        The drifts of each frames of maximum likelihood
    """
    frame_position_indices = np.argmax(membership_probability, axis=0)
    frame_positions = np.array([all_drifts[drifts_in_use[i]] for i in frame_position_indices])
    return frame_positions


def calibrate_drifts_with_reference(
        max_drift: int,
        frame_size: Tuple[int, int],
        model,
        reference,
        drifts_in_use: List[int] = None):
    ec_op = ECOperator(max_drift)
    if drifts_in_use is None:
        drifts_in_use = list(range(ec_op.num_all_drifts))

    expanded_model = ec_op.expand(model, frame_size, drifts_in_use, flatten=True)
    expanded_ref = ec_op.expand(reference, frame_size, drifts_in_use, flatten=True)

    centre_drift_index = max_drift + max_drift * (2 * max_drift + 1)
    if centre_drift_index not in drifts_in_use:
        raise RuntimeError("centre drift is not within the EMC's view. Check the EMC.drift_in_use property")
    idx = drifts_in_use.index(centre_drift_index)
    centre_ref = expanded_ref[idx]  # (N,)

    n1 = np.linalg.norm(centre_ref)
    n2 = np.linalg.norm(expanded_model, axis=1, keepdims=True)  # (M, 1)

    v1 = centre_ref / n1  # (N,)
    v2 = expanded_model / n2  # (M, N)

    diff = np.linalg.norm(v1[None, :] - v2, axis=1)  # (M, )
    recon_drift_centre_index = int(np.argmin(diff))
    recon_centre_drift = ec_op.all_drifts[drifts_in_use[recon_drift_centre_index]]

    calibrating_shift = np.array([max_drift, max_drift]) - recon_centre_drift
    return calibrating_shift


def centre_by_first_frame(frame_positions, max_drift: int, centre_is_origin=True):
    first_frame_position = frame_positions[0]
    calibrating_shift = np.array([max_drift, max_drift]) - first_frame_position
    frame_positions += calibrating_shift
    if centre_is_origin:
        frame_positions -= max_drift
    return calibrating_shift, frame_positions


def centre_by_reference(frame_positions, max_drift: int, frame_size: Tuple[int, int], model, reference, drifts_in_use,
                        centre_is_origin=True):
    calibrating_shift = calibrate_drifts_with_reference(max_drift, frame_size, model, reference, drifts_in_use)
    frame_positions += calibrating_shift
    if centre_is_origin:
        frame_positions -= max_drift
    return calibrating_shift, frame_positions
