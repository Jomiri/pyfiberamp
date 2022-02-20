import unittest
from pyfiberamp.fibers import YbDopedDoubleCladFiber
from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.mode_solver import GaussianMode

class YbDoubleCladTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        yb_number_density = 3e25
        core_r = 5e-6
        background_loss = 0
        length = 3
        pump_cladding_r = 50e-6
        core_to_cladding_ratio = core_r / pump_cladding_r
        core_na = 0.12
        npoints = 20
        tolerance = 1e-5
        cls.input_signal_power = 0.4
        cls.input_pump_power = 47.2

        fiber = YbDopedDoubleCladFiber(length,
                                       core_r, yb_number_density,
                                       background_loss, core_na, core_to_cladding_ratio)
        simulation = SteadyStateSimulation(fiber)
        simulation.add_forward_signal(wl=1030e-9,
                                      input_power=cls.input_signal_power,
                                      mode=GaussianMode(mfd=2 * 4.8e-6, core_radius=fiber.core_radius))
        simulation.add_backward_pump(wl=914e-9, input_power=cls.input_pump_power)
        simulation.set_number_of_nodes(npoints)
        cls.result = simulation.run(tol=tolerance)

    def test_input_signal_power(self):
        simulated_input_power = self.result.powers.forward_signal[0, 0]
        self.assertAlmostEqual(simulated_input_power, self.input_signal_power)

    def test_input_pump_power(self):
        simulated_input_power = self.result.powers.backward_pump[0, -1]
        self.assertAlmostEqual(simulated_input_power, self.input_pump_power)

    def test_output_signal_power(self):
        expected_output_power = 19.685242869832777
        simulated_output_power = self.result.powers.forward_signal[0, -1]
        self.assertAlmostEqual(simulated_output_power, expected_output_power)

    def test_residual_pump_power(self):
        expected_residual_pump_power = 25.24792774963823
        simulated_residual_pump_power = self.result.powers.backward_pump[0, 0]
        self.assertAlmostEqual(simulated_residual_pump_power, expected_residual_pump_power)

    def test_average_excitation(self):
        expected_average_excitation = 0.19820814932170538
        simulated_average_excitation = self.result.overall_average_excitation
        self.assertAlmostEqual(simulated_average_excitation, expected_average_excitation)
