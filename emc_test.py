import unittest
import numpy as np
import matplotlib.pyplot as plt

from emc2d.emc import EMC
from emc2d.transform import Drift
from emc2d.utils import make_drift_vectors
from simtools import build_model, frames_with_random_walk_drifts, random_walk_trajectory


class EmcTestCase(unittest.TestCase):
    def setUp(self):
        img = np.load("data/smiley.npy")
        trj = random_walk_trajectory(max_drift=10, num_steps=200)
        self.model = build_model(img, model_size=(148, 148), mean_count=0.2)
        self.frames = frames_with_random_walk_drifts(self.model, img_size=(128, 128), random_walk_trj=trj, mean_count=0.2)
        self.emc = EMC(self.frames, max_drift=10, init_model=self.model)

    def test_model_frame_dims(self):
        self.assertEqual(self.emc.model_size, (148, 148), "wrong size of initialized model")
        self.assertEqual(self.emc.frame_size, (128, 128), "wrong size of frames")

    def test_expand_compress_closure(self):
        expanded_model = self.emc._expand()
        recon_model = self.emc.compress(expanded_model)
        diff = np.abs(self.model - recon_model)
        self.assertTrue(np.all(diff < 1e-10), "compress cannot reconstruct expanded model")
