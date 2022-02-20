import unittest
import numpy as np
from pyfiberamp.fibers import YbDopedDoubleCladFiber
from pyfiberamp.dynamic import DynamicSimulation
from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.helper_funcs import decibel_to_exp, to_db


class YbDoubleCladTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nt = 1
        r = 3e-6
        cls.signal_loss = decibel_to_exp(1)
        cls.bw_pump_loss = decibel_to_exp(2)
        cls.default_loss = decibel_to_exp(10)
        cls.fiber = YbDopedDoubleCladFiber(length=1, core_radius=r, core_na=0.12, ion_number_density=nt,
                                       ratio_of_core_and_cladding_diameters=0.1, background_loss=cls.default_loss)
        cls.pump_power = 0.5
        cls.signal_power = 0.1
        cls.signal_wl = 1040e-9
        cls.pump_wl = 980e-9
        cls.time_steps = 50000
        cls.z_nodes = 600
        cls.expected_losses = np.array([-1.0, -10.0, -2.0])

    def test_steady_state(self):
        steady_state_simulation = SteadyStateSimulation(self.fiber)
        steady_state_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power, loss=self.signal_loss)
        steady_state_simulation.add_forward_pump(wl=self.pump_wl, input_power=self.pump_power)
        steady_state_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power, loss=self.bw_pump_loss)
        steady_state_result = steady_state_simulation.run(tol=1e-5)
        ss_losses = steady_state_result.powers_at_fiber_end() / np.array([self.signal_power,
                                                                          self.pump_power,
                                                                          self.pump_power])
        self.assertTrue(np.allclose(to_db(ss_losses), self.expected_losses, atol=1e-3, rtol=1e-5))

    def test_dynamic(self):
        dynamic_simulation = DynamicSimulation(self.time_steps, self.fiber)
        dynamic_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power, loss=decibel_to_exp(1))
        dynamic_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power, loss=decibel_to_exp(2))
        dynamic_simulation.add_forward_pump(wl=self.pump_wl, input_power=self.pump_power)
        dynamic_result = dynamic_simulation.run(self.z_nodes, stop_at_steady_state=True, dt=1e-7)
        dyn_losses = dynamic_result.powers_at_fiber_end() / np.array([self.signal_power,
                                                                      self.pump_power,
                                                                      self.pump_power])
        self.assertTrue(np.allclose(to_db(dyn_losses), self.expected_losses, atol=1e-2, rtol=1e-5))

