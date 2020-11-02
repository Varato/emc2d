import numpy as np
import matplotlib.pyplot as plt

from emc2d import emc


# --- Simulated Data ---
intensity = np.load("data/smiley.npy")

plt.imshow(intensity)
plt.show()