import numpy as np
from copy import deepcopy
import unittest

from pyfiberamp.fibers import YbDopedFiber
from pyfiberamp.fibers import YbDopedDoubleCladFiber
from pyfiberamp.dynamic import DynamicSimulation
from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.helper_funcs import decibel_to_exp, to_db
from pyfiberamp.doping_profile import DopingProfile


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
        dynamic_simulation = DynamicSimulation(self.time_steps, self.fiber)
        print('Tested dynamic backends: {}'.format(dynamic_simulation.backends))

    def test_steady_state_python_and_cpp_single_ring(self):
        steady_state_simulation = SteadyStateSimulation(self.fiber)
        steady_state_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power)
        steady_state_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        steady_state_simulation.add_forward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        steady_state_simulation.add_ase(wl_start=1020e-9, wl_end=1040e-9, n_bins=3)
        steady_state_result = steady_state_simulation.run(tol=1e-5)

        dynamic_simulation = DynamicSimulation(self.time_steps, self.fiber)
        dynamic_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power)
        dynamic_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        dynamic_simulation.add_forward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        dynamic_simulation.add_ase(wl_start=1020e-9, wl_end=1040e-9, n_bins=3)

        dynamic_simulation.use_cpp_backend()
        cpp_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_python_backend()
        python_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_pythran_backend()
        pythran_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt,
                                                stop_at_steady_state=True)

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
        fiber_with_rings = deepcopy(self.fiber)
        dynamic_simulation = DynamicSimulation(self.time_steps, fiber_with_rings)
        fiber_with_rings.doping_profile = DopingProfile(ion_number_densities=[self.nt, self.nt],
                                                        radii=[self.fiber.core_radius / 2, self.fiber.core_radius],
                                                        num_of_angular_sections=1,
                                                        core_radius=fiber_with_rings.core_radius)
        dynamic_simulation.add_forward_signal(wl=self.signal_wl, input_power=self.signal_power)
        dynamic_simulation.add_backward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        dynamic_simulation.add_forward_pump(wl=self.pump_wl, input_power=self.pump_power / 2)
        dynamic_simulation.add_ase(wl_start=1020e-9, wl_end=1040e-9, n_bins=3)

        dynamic_simulation.use_cpp_backend()
        cpp_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_python_backend()
        python_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        dynamic_simulation.use_pythran_backend()
        pythran_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt,
                                                stop_at_steady_state=True)

        dynamic_simulation.use_numba_backend()
        numba_result = dynamic_simulation.run(z_nodes=self.z_nodes, dt=self.steady_state_dt, stop_at_steady_state=True)

        cpp_output_powers = cpp_result.powers_at_fiber_end()
        python_output_powers = python_result.powers_at_fiber_end()
        pythran_output_powers = pythran_result.powers_at_fiber_end()
        numba_output_powers = numba_result.powers_at_fiber_end()

        expected_output_regression = np.array([1.24777744e-01, 3.00422189e-01, 2.20328594e-07,
                                               2.32154962e-07, 1.80296310e-07, 3.01232103e-01,
                                               2.42163293e-07, 2.52449606e-07, 1.91762700e-07])
        self.assertTrue(np.allclose(cpp_output_powers, python_output_powers, rtol=1e-6))
        self.assertTrue(np.allclose(cpp_output_powers, expected_output_regression, rtol=1e-6))
        self.assertTrue(np.allclose(pythran_output_powers, expected_output_regression, rtol=1e-6))
        self.assertTrue(np.allclose(numba_output_powers, expected_output_regression, rtol=1e-6))

    """
    def test_steady_state_python_and_cpp_preset_areas_and_overlaps(self):
        dynamic_simulation = DynamicSimulation(self.time_steps, self.fiber)
        fiber_with_rings = deepcopy(self.fiber)
        r = self.fiber.core_radius
        overlaps = [0.5, 0.2]
        fiber_with_rings.doping_profile = DopingProfile(ion_number_densities=[self.nt, self.nt], )
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
    """

    def test_steady_state_reflection(self):
        dynamic_simulation = DynamicSimulation(self.time_steps, self.fiber)
        dynamic_simulation.add_forward_signal(wl=self.signal_wl,
                                              input_power=self.signal_power,
                                              channel_id='forward_signal',
                                              reflection_target_id='reflected_signal',
                                              reflectance=0.04)
        dynamic_simulation.add_backward_signal(wl=self.signal_wl,
                                               input_power=1e-15,
                                               channel_id='reflected_signal')
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
