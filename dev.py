import unittest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

from emc2d import core
from emc2d.sim import build_model, generate_frames

from emc2d.extensions import emc_kernel

np.random.seed(2020)

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


print("run membership probability")

start = time.time()
membership_probability1 = core.compute_membership_probability(emc.frames, model, frame_size, max_drift)
dt1 = time.time() - start
print(membership_probability1.shape)

start = time.time()
membership_probability2 = core.compute_membership_probability_memsaving(emc.frames, model, frame_size, max_drift)
dt2 = time.time() - start
print(membership_probability2.shape)

isclose = np.allclose(membership_probability1, membership_probability2)
print(f"membership probability result same = {isclose}")
print(f"dt1 = {dt1}, dt2 = {dt2}")

print("run merge frames")
start = time.time()
merged_model1 = core.merge_frames_soft(emc.frames, frame_size, model_size, membership_probability1, max_drift)
dt3 = time.time() - start

start = time.time()
merged_model2 = core.merge_frames_soft_memsaving(emc.frames, frame_size, model_size, membership_probability1, max_drift)
dt4 = time.time() - start

# ratio = merged_model1 / merged_model2
diff = np.abs(merged_model2 - merged_model1)

isclose2 = np.allclose(merged_model1, merged_model2, )
print("max error = ", diff.max())
print("average error = ", diff.mean())
# print(f"ratio = ({(ratio.min(), ratio.max())})")
print(f"merge frame result same = {isclose2}")
print(f"dt3 = {dt3}, dt4 = {dt4}")


_, (ax1, ax2, ax3) = plt.subplots(ncols=3)
ax1.imshow(merged_model1)
ax2.imshow(merged_model2)
ax3.imshow(diff)
plt.show()
