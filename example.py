import numpy as np
import matplotlib.pyplot as plt

from emc2d import core


# --- Simulated Data ---
intensity = np.load("data/smiley.npy")

plt.imshow(intensity)
plt.show()