import numpy as np
import matplotlib.pyplot as plt


from emc2d import model, image


a = np.load("smiley.npy")
# a = np.pad(a, pad_width=[100, 50], mode="constant", constant_values=0)

m = model.initialize(init_model=a, max_drift=(50, 50), image_shape=(129, 129))
print("model shape:", m.shape)
indices = range(m.num_drifts)
expanded = m.expand(drift_indices=indices)
composed = expanded.compose()

plt.imshow(composed)
plt.show()
# fig, axes = plt.subplots(ncols=5)
#
# for ax in axes:
#     ax.imshow(next(expanded))
# plt.show()