import unittest
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt

from emc2d import core
from emc2d.sim import build_model, generate_frames

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
            max_drift=self.max_drift,
            init_model=self.model)

    def test_model_frame_dims(self):
        self.assertEqual(self.emc.model_size, (148, 148), "wrong size of initialized model")
        self.assertEqual(self.emc.frame_size, (128, 128), "wrong size of frames")

    def test_membership_probability(self):
        from emc2d.extensions import emc_kernel
        print(emc_kernel.compute_membership_probability)
        print(emc_kernel.merge_frames_into_model)

        expanded_model = self.emc.ec_op.expand(self.model, self.frame_size, flatten=True)
        membership_probability1 = core.compute_membership_probability(expanded_model, self.emc.frames)
        print(membership_probability1.shape)

        membership_probability2 = core.compute_membership_probability_memsaving(
            self.emc.frames, self.model, self.frame_size, self.max_drift)
        print(membership_probability2.shape)
