from typing import Union, Sequence, Iterator, Iterable
import math
import numpy as np
import matplotlib.pyplot as plt


class Image(np.ndarray):
    # see https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for how to subclass ndarray
    # call order:
    #     1. Image.__new__()
    #     2. In Image.__new__(), ndarray.__new__() should be called some how
    #     3. ndarray.__new__() callse __array_finalize__, and then Image.__new__ returns
    #     4. As normal in Python, Image.__init__() (if any) is called after tis __new__() is called
    def __new__(cls, input_array: np.ndarray, info=None):
        # notice: view() calls ndarray.__new__, which in turn calls __array_finalize__()
        if input_array.ndim != 2:
            raise ValueError(f"An Image must be constructed from 2D array, but {input_array.ndim:d}D received")
        obj = np.array(input_array, copy=False).view(cls)
        obj.info = info
        return obj # returns an Image's instance

    def __array_finalize__(self, obj):
        # The obj here can be:
        #     1. None, if Image is explicitly constructed by its constructor
        #     2. Image's instance, if a new image is constructed by slicing an old one
        #     3, Some other type's instance, on which view() has been called.
        if obj is not None:
            self.info = getattr(obj, 'info', None)


class Stack(np.ndarray):
    def __new__(cls, input_array: np.ndarray, batch_dimension=0):
        if input_array.ndim != 3:
            raise ValueError(f"A Stack must be constructed from 3D array, but {input_array.ndim:d}D received")
        obj = np.array(input_array, copy=False).view(cls)
        obj.batch_dimension = batch_dimension
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.batch_dimension = getattr(obj, 'batch_dimension', None)

    def bach_flatten(self):
        return self.reshape(self.shape[self.batch_dimension], -1)


if __name__ == "__main__":
    a = np.arange(12*2).reshape((2,3,4))
    imstk = Stack(a)
    print(imstk)
    print(imstk.bach_flatten())
    # print(im, type(im))
    # print("im.base: ", im.base)
    # plt.imshow(im)
    # plt.show()