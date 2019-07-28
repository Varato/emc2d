from typing import Tuple, Iterable
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


class Image(object):
    def __init__(self, data: np.ndarray):
        if data.ndim > 2:
            raise ValueError(f"An {type(self)} must be constructed from 0D, 1D or 2D array, but {data.ndim:d}D received")
        if data.ndim == 0:
            self._data = np.array(data[np.newaxis][np.newaxis])
        elif data.ndim == 1:
            self._data = np.array(data[np.newaxis, :])
        else:
            self._data = np.array(data)

    def __getitem__(self, item):
        return Image(self._data.__getitem__(item))

    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)

    def __getattr__(self, item):
        # delegates array's attributes and methods, except dunders.
        try:
            return getattr(self._data, item)
        except AttributeError:
            raise AttributeError()

    def crop(self, box: Tuple[int, int, int, int]):
        return self[box[0]:box[1], box[2]:box[3]]

    # binary operations
    def __add__(self, other): return Image(self._data.__add__(other))

    def __sub__(self, other): return Image(self._data.__sub__(other))

    def __mul__(self, other): return Image(self._data.__mul__(other))

    def __matmul__(self, other): return Image(self._data.__matmul__(other))

    def __truediv__(self, other): return Image(self._data.__truediv__(other))

    def __floordiv__(self, other): return Image(self._data.__floordiv__(other))

    def __mod__(self, other): return Image(self._data.__mod__(other))

    def __divmod__(self, other): return Image(self._data.__divmod__(other))

    def __pow__(self, other): return Image(self._data.__pos__(other))

    def __lshift__(self, other): return Image(self._data.__lshift__(other))

    def __rshift__(self, other): return Image(self._data.__rshift__(other))

    def __and__(self, other): return Image(self._data.__and__(other))

    def __or__(self, other): return Image(self._data.__or__(other))

    def __xor__(self, other): return Image(self._data.__xor__(other))

    # comparison

    def __lt__(self, other): return Image(self._data.__lt__(other))

    def __le__(self, other): return Image(self._data.__le__(other))

    def __gt__(self, other): return Image(self._data.__gt__(other))

    def __ge__(self, other): return Image(self._data.__ge__(other))

    def __eq__(self, other): return Image(self._data.__eq__(other))

    def __ne__(self, other): return Image(self._data.__ne__(other))

    # Unary operators

    def __neg__(self): return Image(self._data.__new__())

    def __pos__(self): return Image(self._data.__pos__())

    def __abs__(self): return Image(self._data.__abs__())

    def __invert__(self): return Image(self._data.__invert__())

    def __complex__(self): return Image(self._data.__complex__())

    def __int__(self): return Image(self._data.__int__())

    def __float__(self): return Image(self._data.__float__())

    # Copy

    def __copy__(self): return Image(self._data.__copy__())

    def __deepcopy__(self, memodict={}): return Image(self._data.__deepcopy__(memodict))

    # Others

    def __bool__(self): return self._data.__bool__()

    def __iter__(self): return iter(self._data)


if __name__ == "__main__":
    arr = np.arange(12).reshape(3,4)
    im = Image(arr)
    print(im[0])
    # plt.imshow(im)
    # plt.show()


# class Image(np.ndarray):
#     """
#     An ndarray subclass with dimensionality constrained to 2.
#     Notice:
#         1. Explicit constructing, slicing/indexing an Image object always results in a 2D array-like Image object.
#         2. View-cast from other type with dimensionality other than 2 raises ValueError
#     """
#     def __new__(cls, data):
#         arr = np.asanyarray(data)
#         if arr.ndim > 2:
#             raise ValueError(f"An {cls} must be constructed from 2D array, but {arr.ndim:d}D received")
#         obj = arr.view(cls)
#         if obj.ndim == 0:
#             obj = obj[np.newaxis][np.newaxis]
#         if obj.ndim == 1:
#             obj = obj[np.newaxis, :]
#         return obj
#
#     def __array_finalize__(self, viewed):
#         # called when the object "viewed" has been view-cast to an Image
#         if self.ndim != 2:
#             raise ValueError(f"A {self.ndim:d}D {type(self)} is view-cast by a {type(viewed)}. Expect 2D.")
#
#     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
#         # overriding ufuncs behavior
#         # see https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
#
#         # view-cast any Image in inputs back to numpy array
#         back_inputs = []
#         for i, input_ in enumerate(inputs):
#             if isinstance(input_, Image):
#                 back_inputs.append(input_.view(np.ndarray))
#             else:
#                 back_inputs.append(input_)
#
#         # view-cast any Image in outputs back to numpy array
#         outputs = kwargs.pop('out', None)
#         if outputs:
#             out_args = []
#             for j, output in enumerate(outputs):
#                 if isinstance(output, Image):
#                     out_args.append(output.view(np.ndarray))
#                 else:
#                     out_args.append(output)
#             kwargs['out'] = tuple(out_args)
#         else:
#             outputs = (None,) * ufunc.nout
#
#         results = super(Image, self).__array_ufunc__(ufunc, method, *back_inputs, **kwargs)
#
#         if results is NotImplemented:
#             return NotImplemented
#
#         if ufunc.nout == 1:
#             results = (results,)
#
#         final_results = []
#         for result, output in zip(results, outputs):
#             if output is None:
#                 arr_result = np.asarray(result)
#                 if arr_result.ndim == 2:
#                     final_results.append(arr_result.view(Image))
#                 else:
#                     final_results.append(arr_result)
#             else:
#                 final_results.append(output)
#
#         return final_results[0] if len(final_results) == 1 else final_results
#
#     def __getitem__(self, item):
#         if isinstance(item, int):
#             safe_index = (None, item)
#         elif isinstance(item, slice):
#             safe_index = item
#         elif isinstance(item, tuple) and len(item) == 2:
#             idx, idy = item
#             if all(isinstance(x, int) for x in item):
#                 safe_index = (None, None, idx, idy)
#             elif all(isinstance(x, slice) for x in item):
#                 safe_index = item
#             elif isinstance(idx, int):
#                 safe_index = (None, idx, idy)
#             elif isinstance(idy, int):
#                 safe_index = (idx, idy, None)
#             else:
#                 raise IndexError("Image object is not sliced/indexed properly")
#         else:
#             raise IndexError("Image object is not sliced/indexed properly")
#         return super().__getitem__(safe_index)
#
#     def crop(self, box: Tuple[int, int, int, int]):
#         return self[box[0]:box[1], box[2]:box[3]]



