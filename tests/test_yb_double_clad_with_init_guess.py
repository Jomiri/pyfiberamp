import unittest
import numpy as np

from pyfiberamp.fibers import YbDopedDoubleCladFiber
from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.mode_solver import GaussianMode


class YbDoubleCladWithGuessTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        yb_number_density = 3e25
        core_r = 5e-6
        background_loss = 0
        length = 3
        pump_cladding_r = 50e-6
        core_to_cladding_ratio = core_r / pump_cladding_r
        core_na = 0.12
        tolerance = 1e-5
        cls.input_signal_power = 0.4
        cls.input_pump_power = 47.2

        fiber = YbDopedDoubleCladFiber(length,
                                       core_r, yb_number_density,
                                       background_loss, core_na, core_to_cladding_ratio)
        pump_wavelengths = np.linspace(910, 950, 11) * 1e-9
        cls.gains = []
        init_guess_array = None
        for pump_wl in pump_wavelengths:
            simulation = SteadyStateSimulation(fiber)
            simulation.add_forward_signal(wl=1030e-9,
                                          input_power=cls.input_signal_power,
                                          mode=GaussianMode(mfd=2 * 4.8e-6, core_radius=fiber.core_radius),
                                          channel_id='signal')
            simulation.add_backward_pump(wl=pump_wl, input_power=cls.input_pump_power)
            if init_guess_array is not None:
                simulation.set_guess_array(init_guess_array)
            result = simulation.run(tol=tolerance)
            init_guess_array = result.powers
            result_dict = result.make_result_dict()
            signal_gain = result_dict['signal']['gain']
            cls.gains.append(signal_gain)

    def test_gains(self):
        expected_gains = [16.920274464870143, 16.920807985811383, 16.89380779110342,
                          16.72773065938639, 16.39251181039223, 15.932553278021926,
                          15.327637260825988, 14.58562152909674, 13.963103951187797,
                          13.431642299893685, 12.84524276399409]
        simulated_gains = self.gains
        for expected, simulated in zip(expected_gains, simulated_gains):
            self.assertAlmostEqual(simulated, expected)

