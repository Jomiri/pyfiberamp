import numpy as np


class SlicedArray(np.ndarray):

    def __new__(cls, input_array, slices):
        obj = np.asarray(input_array).view(cls)
        obj.slice_names = set(slices.keys())
        obj.slices = slices
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.slice_names = getattr(obj, 'slice_names', None)
        self.slices = getattr(obj, 'slices', None)

    def item_in_slice_names(self, item):
        return self.__dict__.get('slice_names') is not None and item in self.__dict__.get('slice_names')

    def __getattr__(self, item):
        if self.item_in_slice_names(item):
            return np.array(super().__getitem__([self.slices[item]]))
        else:
            raise AttributeError('SlicedArray does not have attribute {}.'.format(item))

    def __setattr__(self, key, value):
        if self.item_in_slice_names(key):
            super().__setitem__(self.slices[key], value)
        else:
            super().__setattr__(key, value)
