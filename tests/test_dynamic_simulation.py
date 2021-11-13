import numpy as np
from copy import deepcopy
import unittest

from pyfiberamp.fibers import YbDopedFiber
from pyfiberamp.fibers import YbDopedDoubleCladFiber
from pyfiberamp.dynamic import DynamicSimulation
from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.helper_funcs import decibel_to_exp, to_db


class DynamicSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nt = 1e25
        r = 3e-6
        cls.fiber = YbDopedFiber(length=0.1, core_radius=r, core_na=0.12, ion_number_density=cls.nt)
        cls.pump_power = 0.5
        cls.signal_power = 0.1
        cls.signal_wl = 1040e-9
        cls.pump_wl = 980e-9
        cls.time_steps = 50000
        cls.z_nodes = 150
        cls.steady_state_dt = 1e-5

    def test_available_backends(self):
        dynamic_simulation = DynamicSimulation(self.time_steps)
        print('Tested dynamic backends: {}'.format(dynamic_simulation.backends))

    def test_steady_state_python_and_cpp_single_ring(self):
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

        dynamic_simulation.use_pythran_backend()
        pythran_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_numba_backend()
        numba_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        steady_state_output_powers = steady_state_result.powers_at_fiber_end()
        cpp_output_powers = cpp_result.powers_at_fiber_end()
        python_output_powers = python_result.powers_at_fiber_end()
        pythran_output_powers = pythran_result.powers_at_fiber_end()
        numba_output_powers = numba_result.powers_at_fiber_end()
        self.assertTrue(np.allclose(steady_state_output_powers, cpp_output_powers, rtol=1e-3))
        self.assertTrue(np.allclose(cpp_output_powers, python_output_powers, rtol=1e-6))
        self.assertTrue(np.allclose(pythran_output_powers, python_output_powers, rtol=1e-6))
        self.assertTrue(np.allclose(numba_output_powers, python_output_powers, rtol=1e-6))

    def test_steady_state_python_and_cpp_two_rings(self):
        dynamic_simulation = DynamicSimulation(self.time_steps)
        fiber_with_rings = deepcopy(self.fiber)
        fiber_with_rings.set_doping_profile(ion_number_densities=[self.nt, self.nt],
                                            radii=[self.fiber.core_radius/2, self.fiber.core_radius])
        dynamic_simulation.fiber = fiber_with_rings
        dynamic_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power)
        dynamic_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        dynamic_simulation.add_forward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        dynamic_simulation.add_ase(wl_start=1020e-9, wl_end=1040e-9, n_bins=3)

        dynamic_simulation.use_cpp_backend()
        cpp_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_python_backend()
        python_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_pythran_backend()
        pythran_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_numba_backend()
        numba_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        cpp_output_powers = cpp_result.powers_at_fiber_end()
        python_output_powers = python_result.powers_at_fiber_end()
        pythran_output_powers = pythran_result.powers_at_fiber_end()
        numba_output_powers = numba_result.powers_at_fiber_end()

        expected_output_regression = np.array([1.24777656e-01, 3.00423131e-01, 2.20330515e-07,
                                               2.32158298e-07, 1.80295869e-07, 3.01233048e-01,
                                               2.42165526e-07, 2.52453304e-07, 1.91762386e-07])
        self.assertTrue(np.allclose(cpp_output_powers, python_output_powers, rtol=1e-6))
        self.assertTrue(np.allclose(cpp_output_powers, expected_output_regression, rtol=1e-6))
        self.assertTrue(np.allclose(pythran_output_powers, expected_output_regression, rtol=1e-6))
        self.assertTrue(np.allclose(numba_output_powers, expected_output_regression, rtol=1e-6))

    def test_steady_state_python_and_cpp_preset_areas_and_overlaps(self):
        dynamic_simulation = DynamicSimulation(self.time_steps)
        fiber_with_rings = deepcopy(self.fiber)
        r = self.fiber.core_radius
        areas = np.pi * (np.array([r/2, r])**2 - np.array([0, r/2])**2)
        overlaps = [0.5, 0.2]
        fiber_with_rings.set_doping_profile(ion_number_densities=[self.nt, self.nt], areas=areas)
        dynamic_simulation.fiber = fiber_with_rings
        dynamic_simulation.add_forward_signal(wl=self.signal_wl,
                                              input_power=self.signal_power,
                                              mode_shape_parameters={'overlaps': overlaps})
        dynamic_simulation.add_backward_pump(wl=self.pump_wl,
                                             input_power=self.pump_power / 2,
                                             mode_shape_parameters={'overlaps': overlaps})
        dynamic_simulation.add_forward_pump(wl=self.pump_wl,
                                            input_power=self.pump_power / 2,
                                            mode_shape_parameters={'overlaps': overlaps})

        dynamic_simulation.use_cpp_backend()
        cpp_result = dynamic_simulation.run(z_nodes=self.z_nodes,
                                            dt=self.steady_state_dt/10,
                                            stop_at_steady_state=True)

        dynamic_simulation.use_python_backend()
        python_result = dynamic_simulation.run(z_nodes=self.z_nodes,
                                               dt=self.steady_state_dt/10,
                                               stop_at_steady_state=True)

        dynamic_simulation.use_pythran_backend()
        pythran_result = dynamic_simulation.run(z_nodes=self.z_nodes,
                                               dt=self.steady_state_dt/10,
                                               stop_at_steady_state=True)
        dynamic_simulation.use_numba_backend()
        numba_result = dynamic_simulation.run(z_nodes=self.z_nodes,
                                               dt=self.steady_state_dt/10,
                                               stop_at_steady_state=True)

        expected_output_regression = np.array([0.1166232, 0.23989275, 0.23988858])
        cpp_output_powers = cpp_result.powers_at_fiber_end()
        python_output_powers = python_result.powers_at_fiber_end()
        pythran_output_powers = pythran_result.powers_at_fiber_end()
        numba_output_powers = numba_result.powers_at_fiber_end()
        self.assertTrue(np.allclose(cpp_output_powers, python_output_powers, rtol=1e-6))
        self.assertTrue(np.allclose(cpp_output_powers, expected_output_regression, rtol=1e-6))
        self.assertTrue(np.allclose(pythran_output_powers, expected_output_regression, rtol=1e-6))
        self.assertTrue(np.allclose(numba_output_powers, expected_output_regression, rtol=1e-6))

    def test_steady_state_reflection(self):
        dynamic_simulation = DynamicSimulation(self.time_steps)
        dynamic_simulation.fiber = self.fiber
        dynamic_simulation.add_forward_signal(wl=self.signal_wl,
                                              input_power=self.signal_power,
                                              label='forward_signal',
                                              reflection_target='reflected_signal',
                                              reflectance=0.04)
        dynamic_simulation.add_backward_signal(wl=self.signal_wl,
                                               input_power=1e-15,
                                               label='reflected_signal')
        dynamic_simulation.add_backward_pump(wl=self.pump_wl,
                                             input_power=self.pump_power)

        dynamic_simulation.use_cpp_backend()
        cpp_res = dynamic_simulation.run(z_nodes=self.z_nodes,
                                         dt=self.steady_state_dt,
                                         stop_at_steady_state=True)

        dynamic_simulation.use_python_backend()
        python_res = dynamic_simulation.run(z_nodes=self.z_nodes,
                                            dt=self.steady_state_dt,
                                            stop_at_steady_state=True)

        dynamic_simulation.use_pythran_backend()
        pythran_res = dynamic_simulation.run(z_nodes=self.z_nodes,
                                         dt=self.steady_state_dt,
                                         stop_at_steady_state=True)

        dynamic_simulation.use_numba_backend()
        numba_res = dynamic_simulation.run(z_nodes=self.z_nodes,
                                            dt=self.steady_state_dt,
                                            stop_at_steady_state=True)

        cpp_output = cpp_res.powers_at_fiber_end()
        python_output = python_res.powers_at_fiber_end()
        pythran_output = pythran_res.powers_at_fiber_end()
        numba_output = numba_res.powers_at_fiber_end()
        expected_output = np.array([0.11878692, 0.00564412, 0.47651562])
        self.assertTrue(np.allclose(cpp_output, python_output, rtol=1e-6))
        self.assertTrue(np.allclose(cpp_output, expected_output, rtol=1e-6))
        self.assertTrue(np.allclose(pythran_output, expected_output, rtol=1e-6))
        self.assertTrue(np.allclose(numba_output, expected_output, rtol=1e-6))

    def test_per_channel_loss(self):
        nt = 1
        r = 3e-6
        fiber = YbDopedDoubleCladFiber(length=1, core_radius=r, core_na=0.12, ion_number_density=nt,
                                       ratio_of_core_and_cladding_diameters=10, background_loss=decibel_to_exp(10))
        pump_power = 0.5
        signal_power = 0.1
        signal_wl = 1040e-9
        pump_wl = 980e-9
        time_steps = 50000
        z_nodes = 600

        steady_state_simulation = SteadyStateSimulation()
        steady_state_simulation.fiber = fiber
        steady_state_simulation.add_cw_signal(wl=signal_wl, power=signal_power, loss=decibel_to_exp(1))
        steady_state_simulation.add_backward_pump(wl=pump_wl, power=pump_power, loss=decibel_to_exp(2))
        steady_state_simulation.add_forward_pump(wl=pump_wl, power=pump_power)
        steady_state_result = steady_state_simulation.run(tol=1e-5)
        steady_state_result.plot_power_evolution()
        ss_losses = steady_state_result.powers_at_fiber_end() / np.array([signal_power, pump_power, pump_power])

        dynamic_simulation = DynamicSimulation(time_steps)
        dynamic_simulation.fiber = fiber
        dynamic_simulation.add_forward_signal(wl=signal_wl, input_power=signal_power, loss=decibel_to_exp(1))
        dynamic_simulation.add_backward_pump(wl=pump_wl, input_power=pump_power, loss=decibel_to_exp(2))
        dynamic_simulation.add_forward_pump(wl=pump_wl, input_power=pump_power)
        dynamic_result = dynamic_simulation.run(z_nodes, stop_at_steady_state=True, dt=1e-7)
        dynamic_result.plot_power_evolution()
        d_losses = dynamic_result.powers_at_fiber_end() / np.array([signal_power, pump_power, pump_power])

        expected_losses = np.array([-1.0, -10.0, -2.0])

        self.assertTrue(np.allclose(to_db(ss_losses), expected_losses, atol=1e-3, rtol=1e-5))
        self.assertTrue(np.allclose(to_db(d_losses), expected_losses, atol=1e-2, rtol=1e-5))
