import unittest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from emc2d import core
from emc2d.sim import build_model, generate_frames

from emc2d.extensions import emc_kernel


data_path = Path(__file__).resolve().parent / 'test/data/'
img = np.load(data_path/"smiley.npy")

frame_size = (128, 128)
max_drift = 10
mean_count = 1.5
num_frames = 200
motion_sigma = 1.2

model = build_model(img, model_size=(148, 148), mean_count=0.02)
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


expanded_model = emc.ec_op.expand(model, frame_size, flatten=True)
membership_probability1 = core.compute_membership_probability(expanded_model, emc.frames)
print(membership_probability1.shape)

membership_probability2 = core.compute_membership_probability_memsaving(
    emc.frames, model, frame_size, max_drift)
print(membership_probability2.shape)

isclose = np.allclose(membership_probability1, membership_probability2)
print(f"result same = {isclose}")
_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(membership_probability1)
ax2.imshow(membership_probability2)
plt.show()
