from typing import Tuple, Iterable
import numpy as np
import scipy.sparse as sp


class Image(np.ndarray):
    """
    An ndarray subclass with dimensionality constrained to 2.
    Notice:
        1. Explicit constructing, slicing/indexing an Image object always results in a 2D array-like Image object.
        2. View-cast from other type with dimensionality other than 2 raises ValueError
    """
    def __new__(cls, data):
        arr = np.asanyarray(data)
        if arr.ndim > 2:
            raise ValueError(f"An Image must be constructed from 2D array, but {arr.ndim:d}D received")
        obj = arr.view(cls)
        if obj.ndim == 0:
            obj = obj[np.newaxis][np.newaxis]
        if obj.ndim == 1:
            obj = obj[np.newaxis, :]
        return obj

    def __array_finalize__(self, viewed):
        # called when the object "viewed" has been view-cast to an Image
        if self.ndim != 2:
            raise ValueError(f"A {self.ndim:d}D Image is view-cast by a {type(viewed)}. Expect 2D.")

    def crop(self, box: Tuple[int, int, int, int]):
        return self[box[0]:box[1], box[2]:box[3]]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # overriding ufuncs behavior
        # see https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html

        # view-cast any Image in inputs back to numpy array
        back_inputs = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Image):
                back_inputs.append(input_.view(np.ndarray))
            else:
                back_inputs.append(input_)

        # view-cast any Image in outputs back to numpy array
        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Image):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(Image, self).__array_ufunc__(ufunc, method, *back_inputs, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        final_results = []
        for result, output in zip(results, outputs):
            if output is None:
                arr_result = np.asarray(result)
                if arr_result.ndim == 2:
                    final_results.append(arr_result.view(Image))
                else:
                    final_results.append(arr_result)
            else:
                final_results.append(output)

        return final_results[0] if len(final_results) == 1 else final_results

    def __getitem__(self, item):
        if isinstance(item, int):
            safe_index = (None, item)
        elif isinstance(item, slice):
            safe_index = item
        elif isinstance(item, tuple) and len(item) == 2:
            idx, idy = item
            if all(isinstance(x, int) for x in item):
                safe_index = (None, None, idx, idy)
            elif all(isinstance(x, slice) for x in item):
                safe_index = item
            elif isinstance(idx, int):
                safe_index = (None, idx, idy)
            elif isinstance(idy, int):
                safe_index = (idx, idy, None)
            else:
                raise IndexError("Image object is not sliced/indexed properly")
        else:
            raise IndexError("Image object is not sliced/indexed properly")
        return super().__getitem__(safe_index)


class StackIter(object):
    def __init__(self, images: Iterable[Image]):
        self.images = iter(images)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.images)


class StackSparse(sp.csr_matrix):
    pass



