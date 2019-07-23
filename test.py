import numpy as np
import matplotlib.pyplot as plt


from emc2d import model, image


a = np.load("smiley.npy")
m = model.Model(a, max_drift=(10, 10), image_shape=(129, 129))
indices = [0, 1, 2, 3, 4]
expended = m.expand(drift_indices=indices)
# print(next(expended))

fig, axes = plt.subplots(ncols=5)

for ax in axes:
    ax.imshow(next(expended))
plt.show()