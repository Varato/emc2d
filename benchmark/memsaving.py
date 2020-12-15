import os
import psutil
import unittest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

from emc2d import core
from emc2d.sim import build_model, generate_frames

from emc2d.extensions import emc_kernel


def time_and_mem(max_drift: int):
# simulate data
    data_path = Path(__file__).resolve().parent.parent / 'test/data/'
    img = np.load(data_path/"4BED_t40_d5000.npy")

    frame_size = (128, 128)
    max_drift = (max_drift, max_drift)
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

    # pid = os.getpid()
    # ps = psutil.Process(pid)
    # memUse = ps.memory_info()
    # print(pid)
    # print(memUse)

    # initialize EMC
    emc = core.EMC(
        frames=frames,
        frame_size=frame_size,
        max_drift=max_drift,
        init_model=model)

    start = time.time()
    for _ in range(1):
        emc.one_step()
    dt_one_step = time.time() - start

    start = time.time()
    for _ in range(1):
        emc.one_step_memsaving()
    dt_one_step_memsaving = time.time() - start

    return dt_one_step, dt_one_step_memsaving


max_drifts = [2, 5, 10, 15, 20, 30]
t = []
t_memsaving = []
for r in max_drifts:
    print(f"testing for max_drift = {r}")
    t1, t2 = time_and_mem(r)
    t.append(t1)
    t_memsaving.append(t2)

plt.plot(max_drifts, t, 'bv-', label="normal")
plt.plot(max_drifts, t_memsaving, 'ro-', label="memsaving")
plt.ylabel("one step EMC time used")
plt.xlabel("max_drift")
plt.legend()
plt.title("memsaving mode time consuming")
plt.savefig("benchmark.png", dpi=120)
plt.show()