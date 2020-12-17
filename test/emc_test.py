import unittest
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt

from emc2d import core
from emc2d.sim import build_model, generate_frames
from emc2d.extensions import emc_kernel

np.random.seed(2020)


class EmcTestCase(unittest.TestCase):
    def setUp(self):

        data_path = Path('__file__').resolve().parent / 'data/'
        img = np.load(data_path/"smiley.npy")

        self.frame_size = (128, 128)
        self.max_drift = 10
        mean_count = 1.5
        num_frames = 20
        motion_sigma = 1.2

        self.model = build_model(img, model_size=(148, 148), mean_count=0.02)
        self.frames, self.traj = generate_frames(
            self.model,
            window_size=self.frame_size,
            max_drift=self.max_drift,
            num_frames=num_frames,
            mean_count=mean_count,
            motion_sigma=motion_sigma)

        self.emc = core.EMC(
            frames=self.frames,
            frame_size=self.frame_size,
            drift_radius=self.max_drift,
            init_model=self.model)

    def test_model_frame_dims(self):
        self.assertEqual(self.emc.model_size, (148, 148), "wrong size of initialized model")
        self.assertEqual(self.emc.frame_size, (128, 128), "wrong size of frames")

    def test_membership_probability(self):
        start = time.time()
        membership_probability1 = core.compute_membership_probability(emc.frames, model, frame_size, max_drift)
        dt1 = time.time() - start
        print(membership_probability1.shape)

        start = time.time()
        membership_probability2 = core.compute_membership_probability_memsaving(emc.frames, model, frame_size, max_drift)
        dt2 = time.time() - start
        print(membership_probability2.shape)

        isclose = np.allclose(membership_probability1, membership_probability2)
        print(f"result same = {isclose}")
        print(f"dt1 = {dt1}, dt2 = {dt2}")
        self.assertTrue(isclose)
