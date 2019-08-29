import numpy as np
import matplotlib.pyplot as plt
import time


from emc2d import model
from emc2d.sim_tools import random_walk_trajectory, frames_with_random_walk_drifts
from emc2d.frame_stack import FrameStack
from emc2d.main import emc_simple

from emc2d.fn_tools import take


a = np.load("smiley.npy")
a = np.pad(a, pad_width=[100, 100], mode="constant", constant_values=0)

m = model.initialize(init_model=a, max_drift=(10, 10), image_shape=(129, 129))
print("model shape:", m.model_shape)

rw = random_walk_trajectory(m.max_drift, 200)
frames = frames_with_random_walk_drifts(m, rw, mean_count=1.2)

# ======

frame_stack = FrameStack(frames, m.image_shape)

emc = emc_simple(frame_stack, m.drift_setup)

tic = time.time()
results = take(30, emc)
toc = time.time()
print("time spent:", toc-tic)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(frames.sum(0))
ax2.imshow(results[-1][0].content)
plt.show()


