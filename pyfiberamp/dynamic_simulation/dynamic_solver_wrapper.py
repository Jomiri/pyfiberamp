from abc import ABC, abstractmethod

from pyfiberamp.helper_funcs import *
from pyfiberamp import SimulationResult
import numpy as np
from collections import namedtuple

DEFAULT_GROUP_INDEX = 1.45

Reflection = namedtuple('Reflection', ('source', 'target', 'R'))
solution = namedtuple('Solution', ('x', 'y'))


class DynamicSolverBase(ABC):
    def __init__(self, channels, fiber, n_nodes, max_iterations, dt='auto', P=None, N2=None,
                 reflections=None, P_in_out=None):
        self.channels = channels
        self.fiber = fiber
        self.reflections = self.check_reflection(reflections)
        self.simulation_array_shape = (channels.number_of_channels, n_nodes)
        self.max_iterations = max_iterations
        self.P = self.check_P(P, self.simulation_array_shape)
        self.N2 = self.check_N2(N2, self.simulation_array_shape)
        self.P_in_out = self.check_P_in_out(P_in_out)
        self.reflections = self.check_reflection(reflections)
        self.dz = fiber.length / (n_nodes - 1)
        self.dt = self.check_dt(dt)
        self.z = np.linspace(0, fiber.length, n_nodes)
        self.t = np.linspace(0, max_iterations, max_iterations) * self.dt

        self.stop_at_steady_state = False
        self.steady_state_tolerance = 1e-4

    def check_reflection(self, reflections):
        if reflections is None:
            return []
        else:
            return self._check_reflection_labels(reflections)

    def _check_reflection_labels(self, reflections):
        processed = []
        for r in reflections:
            source_index = self.channels.get_channel_index(r.source)
            target_index = self.channels.get_channel_index(r.target)
            R = r.R
            assert(0 <= R <= 1)
            processed.append((source_index, target_index, R))
        return processed

    def check_P(self, P, simulation_array_shape):
        if P is None:
            P = np.ones(simulation_array_shape) * SIMULATION_MIN_POWER
        else:
            assert(P.shape == simulation_array_shape)
        return P

    def check_N2(self, N2, simulation_array_shape):
        if N2 is None:
            N2 = np.zeros(simulation_array_shape[1])
        else:
            assert(N2.shape == (1, simulation_array_shape[1]))
        return N2

    def check_P_in_out(self, P_in_out):
        if P_in_out is None:
            P_in = self.channels.get_input_powers()
            P_in_out = np.tile(P_in[:, np.newaxis], (1, self.max_iterations + 1))
        else:
            assert P_in_out.shape == (self.simulation_array_shape[0], self.max_iterations + 1)
        return P_in_out

    def check_dt(self, dt):
        if dt == 'auto':
            cn = c / DEFAULT_GROUP_INDEX
            dt = self.dz / cn
        else:
            assert(dt < self.fiber.spectroscopy.upper_state_lifetime)
        return dt

    def run(self):
        a = self.channels.get_absorption()
        g = self.channels.get_gain()
        l = self.channels.get_background_loss()
        v = self.channels.get_frequencies()
        dv = self.channels.get_frequency_bandwidths()
        directions = self.channels.get_propagation_directions()
        n_forward = np.sum(directions[directions == 1])

        n_iter = self.solve(self.P, self.N2, g, a, l, v, dv, self.P_in_out, self.reflections,
                                    self.fiber.core_r, self.fiber.spectroscopy.upper_state_lifetime,
                                    self.fiber.length, self.fiber.ion_number_density, n_forward,
                                    self.steady_state_tolerance, self.dt, self.stop_at_steady_state)

        sol = solution(x=self.z, y=self.P)
        return sol

    @abstractmethod
    def solve(self):
        pass


class DynamicSolverCpp(DynamicSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from fiber_simulation_pybindings import simulate


    def solve(self, *args):


    def run_cpp_simulation(self, channels, fiber, n_nodes, P_in_out=None, P=None, reflection=None, dt='auto'):



        res = SimulationResult(solution=sol,
           upper_level_fraction=N2/Nt,
           slices=channels.get_slices(),
           wavelengths=channels.get_wavelengths(),
           is_passive_fiber=fiber.is_passive_fiber())
        #res.use_db_scale = True
