import unittest
from pyfiberamp import FiberAmplifierSimulationWithRaman, decibel_to_exp
from pyfiberamp.fibers import PassiveFiber
import numpy as np


class PassiveRamanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core_r = np.sqrt(50e-12 / np.pi)
        background_loss = decibel_to_exp(0.0002)
        length = 20000
        core_NA = 0.12
        fiber = PassiveFiber(length, core_r, background_loss, core_NA)
        fiber.effective_area_type = 'core_area'
        simulation = FiberAmplifierSimulationWithRaman(fiber)
        simulation.add_cw_signal(wl=1550e-9, power=0.600)
        simulation.add_raman(backward_raman_allowed=False)

        npoints = 20
        cls.result = simulation.run(npoints, tol=1e-10)
        assert(cls.result.success())

    def test_remaining_signal_power(self):
        expected_remaining_signal_power = 0.134685392014
        simulated_remaining_signal_power = self.result.forward_signals[0, -1]
        self.assertAlmostEqual(simulated_remaining_signal_power, expected_remaining_signal_power)

    def test_generated_forward_raman_power(self):
        expected_forward_raman_power = 0.0970689904605
        simulated_forward_raman_power = self.result.forward_ramans[0, -1]
        self.assertAlmostEqual(simulated_forward_raman_power, expected_forward_raman_power)


