import unittest

import numpy as np
from pyfiberamp.fibers import YbDopedDoubleCladFiber, DopingProfile
from pyfiberamp.dynamic import DynamicSimulation
from pyfiberamp.mode_solver import default_mode_solver as mode_solver


class ConfinedFiberTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yb_number_density = 5e25
        radii = np.array([2, 4, 6, 8, 10]) * 1e-6
        core_radius = 10e-6
        doping_radius = 6e-6
        confined_doping_profile = DopingProfile(radii=radii[:-2],
                                                num_of_angular_sections=6,
                                                ion_number_densities=[yb_number_density] * 18,
                                                core_radius=core_radius)

        confined_fiber_length = 30
        core_na = 0.065
        core_clad_ratio = 20 / 400
        signal_wl = 1064e-9
        pump_wl = 976e-9
        confined_fiber_20_400 = YbDopedDoubleCladFiber(length=confined_fiber_length,
                                                       core_radius=core_radius,
                                                       ion_number_density=0,
                                                       background_loss=0,
                                                       core_na=core_na,
                                                       ratio_of_core_and_cladding_diameters=core_clad_ratio)
        confined_fiber_20_400.doping_profile = confined_doping_profile
        fundamental_mode_content = 0.98
        lp11_content = (1 - fundamental_mode_content)
        signal_power = 5  # W
        LP01_power = fundamental_mode_content * signal_power
        LP11_cos_power = 0.5 * lp11_content * signal_power
        LP11_sin_power = 0.5 * lp11_content * signal_power
        pump_power = 1000
        max_time_steps = 1_000_000
        z_nodes = 200
        fiber_modes = mode_solver.find_all_modes(core_radius, core_na, signal_wl)
        confined_fiber_simulation = DynamicSimulation(max_time_steps, confined_fiber_20_400)
        confined_fiber_simulation.add_forward_signal(wl=signal_wl,
                                                     input_power=LP01_power,
                                                     mode=fiber_modes[0],
                                                     channel_id='LP01')
        confined_fiber_simulation.add_forward_signal(wl=signal_wl,
                                                     input_power=LP11_cos_power,
                                                     mode=fiber_modes[1],
                                                     channel_id='LP11_cos')
        confined_fiber_simulation.add_forward_signal(wl=signal_wl,
                                                     input_power=LP11_sin_power,
                                                     mode=fiber_modes[2],
                                                     channel_id='LP11_sin')

        confined_fiber_simulation.add_backward_pump(wl=pump_wl, input_power=pump_power, channel_id='Pump')
        cls.confined_res = confined_fiber_simulation.run(z_nodes, dt=1e-7, stop_at_steady_state=True)

    def test_output_power(self):
        self.assertAlmostEqual(self.confined_res.powers[0, -1], 879.2806523068698)

    def test_inversion(self):
        expected = np.array([0.05941684, 0.05941684, 0.05941684, 0.05941684, 0.05941684,
                             0.05941684, 0.06602226, 0.06602226, 0.06602226, 0.06602226,
                             0.06602226, 0.06602226, 0.08252444, 0.08252444, 0.08252444,
                             0.08252444, 0.08252444, 0.08252444])
        self.assertTrue(np.allclose(self.confined_res.upper_level_fraction[:, 1], expected))
