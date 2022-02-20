from abc import ABC, abstractmethod

from pyfiberamp.helper_funcs import *
from pyfiberamp.dynamic import DynamicSimulationResult
import numpy as np


class DynamicSolverBase(ABC):
    def __init__(self, channels, fiber, n_nodes, max_iterations, dt, P, N2,
                 stop_at_steady_state, steady_state_tolerance, convergence_checking_interval):
        self.channels = channels
        self.fiber = fiber
        self.reflections = self.channels.get_reflections()
        self.simulation_array_shape = (channels.number_of_channels, n_nodes)
        self.max_iterations = max_iterations
        self.P = self._check_P(P, self.simulation_array_shape)
        self.N2 = self._check_N2(N2, self.simulation_array_shape, self.fiber.num_ion_populations)
        self.P_in_out = self.channels.get_dynamic_input_powers(max_iterations)
        self.dz = fiber.length / (n_nodes - 1)
        self.dt = self._check_dt(dt)
        self.z = np.linspace(0, fiber.length, n_nodes)
        self.t = np.linspace(0, max_iterations, max_iterations) * self.dt

        self.stop_at_steady_state = stop_at_steady_state
        self.steady_state_tolerance = steady_state_tolerance
        self.convergence_checking_interval = convergence_checking_interval

    def _check_P(self, P, simulation_array_shape):
        if P is None:
            P = np.ones(simulation_array_shape) * SIMULATION_MIN_POWER
        else:
            assert (P.shape == simulation_array_shape)
        return P

    def _check_N2(self, N2, simulation_array_shape, num_ion_populations):
        n2_shape = (num_ion_populations, simulation_array_shape[1])
        if N2 is None:
            N2 = np.zeros(n2_shape)
        else:
            assert (N2.shape == n2_shape)
        return N2

    def _check_dt(self, dt):
        if dt == 'auto':
            cn = c / self.fiber.core_refractive_index
            dt = self.dz / cn
        else:
            assert (dt < self.fiber.spectroscopy.upper_state_lifetime)
        return dt

    def run(self):
        a = self.channels.get_absorption()
        g = self.channels.get_gain()
        l = self.channels.get_background_loss()
        v = self.channels.get_frequencies()
        dv = self.channels.get_frequency_bandwidths()
        directions = self.channels.get_propagation_directions()
        n_forward = int(np.sum(directions[directions == 1]))

        n_channels = self.channels.number_of_channels
        ion_number_densities = np.tile(self.fiber.doping_profile.ion_number_densities, n_channels)
        ion_cross_section_areas = np.tile(self.fiber.doping_profile.areas, n_channels)

        n_iter = self.solve(self.P, self.N2, g, a, l, v, dv, self.P_in_out, self.reflections,
                            ion_cross_section_areas, self.fiber.spectroscopy.upper_state_lifetime,
                            self.fiber.length, ion_number_densities, n_forward,
                            self.steady_state_tolerance, self.dt, self.stop_at_steady_state, n_channels,
                            self.convergence_checking_interval)

        P_out = self.P_in_out[:, :n_iter]
        self.N2 = self.extrapolate_first_point(self.N2)
        res = DynamicSimulationResult(z=self.z,
                                      t=self.t[:n_iter],
                                      powers=self.P,
                                      upper_level_fraction=self.N2 / self.fiber.doping_profile.ion_number_densities[:,
                                                                     np.newaxis],
                                      output_powers=P_out,
                                      channels=self.channels,
                                      fiber=self.fiber)
        return res

    def extrapolate_first_point(self, N2):
        N2[:, 0] = 2 * N2[:, 1] - N2[:, 2]
        return N2

    @abstractmethod
    def solve(self, *args):
        pass
