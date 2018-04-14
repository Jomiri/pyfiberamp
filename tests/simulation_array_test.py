import unittest
import numpy as np
from copy import deepcopy

from pyfiberamp import SlicedArray


class SimulationArrayTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        arr = np.linspace(1, 16, 16)
        cls.arr_2d = np.reshape(arr, (4,4))
        cls.arr_1d = cls.arr_2d[0,:]
        slices = {'forward_signal': slice(0, 1),
                  'backward_signal': slice(1, 1),
                  'forward_pump': slice(1, 2),
                  'backward_pump': slice(2, 4),
                  'forward_ase': slice(4, 4),
                  'backward_ase': slice(4, 4),
                  'forward_raman': slice(4, 4),
                  'backward_raman': slice(4, 4)}
        cls.sim_arr_1d = SlicedArray(cls.arr_1d, slices)
        cls.sim_arr_2d = SlicedArray(cls.arr_2d, slices)

    def test_arr(self):
        self.assertTrue(True)
        print(self.arr_2d)

    def test_sim_arr_2d(self):
        self.assertTrue(True)
        print(self.sim_arr_2d.forward_signal)
        print(self.sim_arr_2d.backward_pump)

    def test_sim_arr_1d(self):
        self.assertTrue(True)
        print(self.sim_arr_1d.forward_signal)

    def test_substitution_arr_1d(self):
        new_arr_1d = deepcopy(self.sim_arr_1d)
        print(type(new_arr_1d))
        new_arr_1d.forward_pump = 7
        print('subst test')
        print(new_arr_1d)
        print(new_arr_1d.slices)
        self.assertTrue(True)