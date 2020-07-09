import numpy as np
import matplotlib.pyplot as plt


from emc2d.emc import EMC
from emc2d.transform import Drift
from emc2d.utils import make_drift_vectors
from simtools import build_model, frames_with_random_walk_drifts, random_walk_trajectory


img = np.load("data/smiley.npy")

model = build_model(img, model_size=(148, 148), mean_count=0.2)
trj = random_walk_trajectory(max_drift=10, num_steps=200)
frames = frames_with_random_walk_drifts(model, img_size=(128, 128), random_walk_trj=trj, mean_count=0.2)

emc = EMC(frames, max_drift=10, init_model=model)

expanded_model = emc.expand(model)
model_recon = emc.compress(expanded_model)
diff = np.abs(model - model_recon)
print("diff:", np.max(diff))

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
ax1.imshow(model)
ax2.imshow(is_zero)
ax3.imshow(model-model_recon)
plt.show()

# print(frames.mean())

# frames_sum = frames.sum(0)

# fig, axes = plt.subplots(ncols=6)


# for i, ax in enumerate(axes):
#     if i == 0:
#         ax.imshow(expanded_model.sum(0))
#     else:
#         ax.imshow(expanded_model[(i-1)*6])
#     ax.axis(False)

# plt.show()

