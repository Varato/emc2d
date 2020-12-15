import unittest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

from emc2d import core
from emc2d.sim import build_model, generate_frames

from emc2d.extensions import emc_kernel


data_path = Path(__file__).resolve().parent / 'test/data/'
img = np.load(data_path/"4BED_t40_d5000.npy")

frame_size = (128, 128)
max_drift = (15, 15)
mean_count = 1.5
num_frames = 500
motion_sigma = 1.2
model_size = tuple(frame_size[d] + 2 * max_drift[d] for d in [0,1])

model = build_model(img, model_size=model_size, mean_count=0.02)
frames, traj = generate_frames(
    intensity=model,
    window_size=frame_size,
    max_drift=max_drift,
    num_frames=num_frames,
    mean_count=mean_count,
    motion_sigma=motion_sigma)

emc = core.EMC(
    frames=frames,
    frame_size=frame_size,
    max_drift=max_drift,
    init_model=model)

start = time.time()
expanded_model = emc.ec_op.expand(model, frame_size, flatten=True)
membership_probability1 = core.compute_membership_probability(expanded_model, emc.frames)
dt1 = time.time() - start
print(membership_probability1.shape)

start = time.time()
membership_probability2 = core.compute_membership_probability_memsaving(
    emc.frames, model, frame_size, max_drift)
dt2 = time.time() - start
print(membership_probability2.shape)

isclose = np.allclose(membership_probability1, membership_probability2)
print(f"result same = {isclose}")
print(f"dt1 = {dt1}, dt2 = {dt2}")
_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(membership_probability1)
ax2.imshow(membership_probability2)
plt.show()
