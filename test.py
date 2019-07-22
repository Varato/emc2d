import numpy as np
import matplotlib.pyplot as plt

from emc2d import model, utils

a = np.load("smiley.npy")
print(a.shape)
# plt.imshow(a)
drift_setup = utils.DriftSetup(max_drift=(10,10), img_size=(129,129))
model = model.Model(init_model=a, mean=0.5, drift_setup=drift_setup)

expanded = list(model.expand(range(len(drift_setup.drift_table))))

composed = model.compose(iter(expanded), drift_indices=range(len(drift_setup.drift_table)))
plt.imshow(model._data)
print(model._data.shape)
fig, axes = plt.subplots(ncols=5)
for i, ax in enumerate(axes):
    ax.imshow(expanded[i])
plt.show()

print(abs(model._data - a).min())