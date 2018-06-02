import numpy as np
import unittest
from pyfiberamp.fibers import YbDopedFiber
from pyfiberamp.dynamic import DynamicSimulation
from pyfiberamp.steady_state import SteadyStateSimulation


class DynamicSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nt = 1e25
        r = 3e-6
        cls.fiber = YbDopedFiber(length=0.1, core_radius=r, core_na=0.12, ion_number_density=nt)
        cls.pump_power = 0.5
        cls.signal_power = 0.1
        cls.signal_wl = 1040e-9
        cls.pump_wl = 980e-9
        cls.time_steps = 50000
        cls.z_nodes = 150
        cls.steady_state_dt = 1e-5


    def test_steady_state_python_and_cpp(self):
        steady_state_simulation = SteadyStateSimulation()
        steady_state_simulation.fiber = self.fiber
        steady_state_simulation.add_cw_signal(wl=self.signal_wl, power=self.signal_power)
        steady_state_simulation.add_backward_pump(wl=self.pump_wl, power=self.pump_power/2)
        steady_state_simulation.add_forward_pump(wl=self.pump_wl, power=self.pump_power/2)
        steady_state_simulation.add_ase(wl_start=1020e-9, wl_end=1040e-9, n_bins=3)
        steady_state_result = steady_state_simulation.run(tol=1e-5)

        dynamic_simulation = DynamicSimulation(self.time_steps)
        dynamic_simulation.fiber = self.fiber
        dynamic_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power)
        dynamic_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power/2)
        dynamic_simulation.add_forward_pump(wl=self.pump_wl, input_power=self.pump_power/2)
        dynamic_simulation.add_ase(wl_start=1020e-9, wl_end=1040e-9, n_bins=3)

        dynamic_simulation.use_cpp_backend()
        cpp_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_python_backend()
        python_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        steady_state_output_powers = steady_state_result.powers_at_fiber_end()
        cpp_output_powers = cpp_result.powers_at_fiber_end()
        python_output_powers = python_result.powers_at_fiber_end()
        self.assertTrue(np.allclose(steady_state_output_powers, cpp_output_powers, rtol=1e-3))
        self.assertTrue(np.allclose(cpp_output_powers, python_output_powers, rtol=1e-6))

    def test_steady_state_reflection(self):
        dynamic_simulation = DynamicSimulation(self.time_steps)
        dynamic_simulation.fiber = self.fiber
        dynamic_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power, label='forward_signal',
                                              reflection_target='reflected_signal', reflectance=0.04)
        dynamic_simulation.add_backward_signal(wl=self.signal_wl, input_power=1e-15, label='reflected_signal')
        dynamic_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power)

        dynamic_simulation.use_cpp_backend()
        cpp_res = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_python_backend()
        python_res = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        cpp_output = cpp_res.powers_at_fiber_end()
        python_output = python_res.powers_at_fiber_end()
        expected_output = np.array([0.1122059, 0.00503606, 0.48387128])
        self.assertTrue(np.allclose(cpp_output, expected_output, rtol=1e-6))
        self.assertTrue(np.allclose(cpp_output, python_output, rtol=1e-6))
