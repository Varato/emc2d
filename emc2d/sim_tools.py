"""
This module defines functions for drift-correction simulations.
i.e. these functions are usefull only we have the true model
"""
from typing import Tuple, Iterable
import numpy as np
import logging
import random

from .fn_tools import iterate, take
from .model import Model
from .transforms import expand, compress


logger = logging.getLogger("processing.emc_motioncorr.simulation_tools")


def random_walk_trajectory(max_drift: Tuple[int, int],
                           num_steps: int,
                           start: Tuple[int, int] = (0, 0), rand_seed=None):
    """
    generate translational vectors for constrained random walk (RW).

    """
    random.seed(rand_seed)

    def one_step(r: Tuple[int, int]):
        dr = (random.randint(-1, 1), random.randint(-1, 1))
        return tuple(r[d]-dr[d] if abs(r[d]+dr[d]) > max_drift[d] else r[d]+dr[d] for d in (0, 1))

    drifts = take(num_steps, iterate(one_step, start))
    return drifts


def frames_with_random_walk_drifts(model: Model, drifts: Iterable[Tuple[int, int]], mean_count, rand_seed=None):
    """
    """
    np.random.seed(rand_seed)
    drift_indices = model.drift_setup.get_drift_indices(drifts)
    patterns = np.array([mean_count * p / p.mean() for p in expand(model, drift_indices)])
    frames = np.random.poisson(np.clip(patterns, 1e-7, None)).astype(np.int)
    return frames


# def bound_drifts(r: Tuple[float, float], dr: Tuple[float, float], bounds: Tuple[float, float, float, float]):
#     def in_bounds(r, bounds):
#         return bounds[0] <= r[0] <= bounds[1] and bounds[2] <= r[1] <= bounds[3]
#
#     def y_intersect(x, r0, r1):
#         if r1[0] == r0[0]:
#             return None
#         return (x-r0[0])/(r1[0]-r0[0]) * (r1[1]-r0[1]) + r0[1]
#
#     def x_intersect(y, r0, r1):
#         if r1[1] == r0[1]:
#             return None
#         return (y-r0[1])/(r1[1]-r0[1]) * (r1[0]-r0[0]) + r0[0]
#
#     if not in_bounds(r, bounds):
#         raise ValueError('r is not in the boundary')
#
#     r1 = tuple(r[d] + dr[d] for d in (0,1))
#     if in_bounds(r1, bounds):
#         return r1
#
#     possible_y = bounds[3] if dr[1] > 0 else bounds[2]
#     possible_x = bounds[1] if dr[0] > 0 else bounds[0]
#     x_at_boundary = x_intersect(possible_y, r, r1)
#     y_at_boundary = y_intersect(possible_x, r, r1)
#
#     if x_at_boundary is None:
#         intersect_point = (possible_x, y_at_boundary)
#         mirror = (-1, 1)
#     elif y_at_boundary is None:
#         intersect_point = (x_at_boundary, possible_y)
#         mirror = (1,-1)
#     else:
#         if (bounds[0] <= x_at_boundary <= bounds[1]) and not (bounds[2] <= y_at_boundary <= bounds[3]):
#             intersect_point = (x_at_boundary, possible_y)
#             mirror = (1, -1)
#         elif not (bounds[0] <= x_at_boundary <= bounds[1]) and (bounds[2] <= y_at_boundary <= bounds[3]):
#             intersect_point = (possible_x, y_at_boundary)
#             mirror = (-1, 1)
#         else:
#             intersect_point = (x_at_boundary, y_at_boundary)
#             mirror = (-1, -1)
#
#     reflect = tuple((r1[d] - intersect_point[d]) * mirror[d] for d in (0, 1))
#     # new_dr = reflect + intersect_point - r0
#     return bound_drifts(intersect_point, reflect, bounds)

