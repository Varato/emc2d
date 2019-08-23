import numpy as np
import matplotlib.pyplot as plt


from emc2d import model, image
from emc2d import transformations as tf


a = np.load("smiley.npy")

a = np.pad(a, pad_width=[100, 100], mode="constant", constant_values=0)

m = model.initialize(init_model=a, max_drift=(50, 50), image_shape=(129, 129))
indices = range(m.num_drifts)

print("model shape:", m.content.shape)

comp = tf.compose(tf.expand(m, indices))

plt.imshow(comp.content)
plt.show()
