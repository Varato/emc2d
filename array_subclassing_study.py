from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import operator


def make_bin_op(oper ):
    def op(self, other):
        if isinstance(other, Image):
            return Image(oper(self._data, other._data))
        else:
            return Image(oper(self._data, other))
    return op


class Image(object):
    def __init__(self, data: np.ndarray):
        self._data = np.array(data)

    def __getitem__(self, item):
        return self._data.__getitem__(item)

    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)

    def __getattr__(self, item):
        # delegates array's attributes and methods, except dunders.
        try:
            return getattr(self._data, item)
        except AttributeError:
            raise AttributeError()



im = np.arange(12).reshape(3,4).view(Image)


plt.imshow(im)
plt.show()