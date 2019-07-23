from typing import Tuple, Iterable
import numpy as np


class Image(np.ndarray):
    # see https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for how to subclass ndarray
    # call order:
    #     1. Image.__new__()
    #     2. In Image.__new__(), ndarray.__new__() should be called some how
    #     3. ndarray.__new__() calls __array_finalize__, and then Image.__new__ returns
    #     4. As normal in Python, Image.__init__() (if any) is called after tis __new__() is called
    def __new__(cls, input_array: np.ndarray, info=None):
        # notice: view() calls ndarray.__new__, which in turn calls __array_finalize__()
        if input_array.ndim != 2:
            raise ValueError(f"An Image must be constructed from 2D array, but {input_array.ndim:d}D received")
        obj = np.array(input_array, copy=False).view(cls)
        obj.info = info
        return obj  # returns an Image's instance

    def __array_finalize__(self, viewed):
        # This method is called when (just before?) numpy constructs a new array.
        # obj here is the object where the new array has been constructed from
        # obj is None if it's been constructed from no where.
        if viewed is None:
            return
        self.info = getattr(viewed, 'info', None)

    def crop(self, box: Tuple[int, int, int, int]):
        return self[box[0]:box[1], box[2]:box[3]]


class StackIter(object):
    def __init__(self, images: Iterable[Image]):
        self.images = iter(images)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.images)


class StackCompact(np.ndarray):
    """
    dimension 0 is considered as batch dimension
    """
    def __new__(cls, input_array: np.ndarray):
        if input_array.ndim != 3:
            raise ValueError(f"A Stack must be constructed from 3D array, but {input_array.ndim:d}D received")
        obj = np.array(input_array, copy=False).view(cls)
        return obj

    def __array_finalize__(self, viewed):
        if viewed is None:
            return
        if viewed.ndim != 3:
            raise ValueError(f"A Stack must be constructed from 3D array, but {viewed.ndim:d}D received")


class StackSparse:
    pass



