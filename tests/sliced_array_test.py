import unittest
import numpy as np
from copy import deepcopy

from pyfiberamp.util.sliced_array import SlicedArray


class SimulationArrayTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        arr = np.linspace(1, 16, 16)
        array_2d = np.reshape(arr, (4,4))
        array_1d = array_2d[0,:]
        slices = {'s01': slice(0, 1),
                  's11': slice(1, 1),
                  's12': slice(1, 2),
                  's24': slice(2, 4),
                  's44': slice(4, 4)}

        cls.sliced_array_1d = SlicedArray(array_1d, slices)
        cls.sliced_array_2d = SlicedArray(array_2d, slices)

    def test_slicing_1d_array(self):
        expected_s01 = np.array([1])
        expected_s12 = np.array([2])
        expected_s24 = np.array([3, 4])
        self.assertTrue(np.array_equal(self.sliced_array_1d.s01, expected_s01))
        self.assertTrue(self.sliced_array_1d.s11.size == 0)
        self.assertTrue(np.array_equal(self.sliced_array_1d.s12, expected_s12))
        self.assertTrue(np.array_equal(self.sliced_array_1d.s24, expected_s24))
        self.assertTrue(self.sliced_array_1d.s44.size == 0)

    def test_slicing_2d_array(self):
        expected_s01 = np.array([[1, 2, 3, 4]])
        expected_s12 = np.array([[5, 6, 7, 8]])
        expected_s24 = np.array([[9, 10, 11, 12],
                                 [13, 14, 15, 16]])
        self.assertTrue(np.array_equal(self.sliced_array_2d.s01, expected_s01))
        self.assertTrue(self.sliced_array_2d.s11.size == 0)
        self.assertTrue(np.array_equal(self.sliced_array_2d.s12, expected_s12))
        self.assertTrue(np.array_equal(self.sliced_array_2d.s24, expected_s24))
        self.assertTrue(self.sliced_array_2d.s44.size == 0)

    def test_substitution_1d_array(self):
        new_array_1d = deepcopy(self.sliced_array_1d)
        new_array_1d.s01 = 0
        new_array_1d.s11 = 1e8 # empty substitution
        new_array_1d.s12 = 10
        new_array_1d.s24 = [8, 9]
        expected = np.array([0, 10, 8, 9])
        self.assertTrue(np.array_equal(new_array_1d, expected))

    def test_substitution_2d_array(self):
        new_array_2d = deepcopy(self.sliced_array_2d)
        new_array_2d.s01 = [8, 8, 9, 9]
        new_array_2d.s24 = np.array([[0, 0, 0, 0],
                                    [1, 1, 1, 1]])
        expected = np.array([[8, 8, 9, 9],
                             [5, 6, 7, 8],
                             [0, 0, 0, 0],
                             [1, 1, 1, 1]])
        self.assertTrue(np.array_equal(new_array_2d, expected))

    def test_failing_substitution_1d_array(self):
        new_array_1d = deepcopy(self.sliced_array_1d)
        with self.assertRaises(ValueError):
            new_array_1d.s44 = [13, 14]

        with self.assertRaises(ValueError):
            new_array_1d.s01 = [1, 5]


