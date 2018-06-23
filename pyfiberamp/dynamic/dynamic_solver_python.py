import numpy as np
from pyfiberamp.dynamic.dynamic_solver_base import DynamicSolverBase
from pyfiberamp.dynamic.dynamic_solver_util import *
from pyfiberamp.dynamic.dynamic_boundary_conditions import DynamicBoundaryConditions
from pyfiberamp.dynamic.convergence_checker import ConvergenceChecker
from pyfiberamp.helper_funcs import *


class DynamicSolverPython(DynamicSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, P, N2, g, a, l, v, dv, P_in_out, reflections,
              ion_cross_section_areas, upper_state_lifetime,
              fiber_length, ion_number_densities, n_forward,
              steady_state_tolerance, dt, stop_at_steady_state, n_channels,
              convergence_checking_interval):
        nodes = P.shape[1]
        max_iterations = P_in_out.shape[1] - 1
        dz = fiber_length / (nodes - 1)
        n_ion_populations = int(len(ion_cross_section_areas) / n_channels)
        channel_params = ChannelParameters(a, g, l, v, dv, nodes, ion_cross_section_areas, ion_number_densities, n_channels)
        boundary_conditions = DynamicBoundaryConditions(P_in_out, reflections, n_forward)
        dn2dt = dNdT(channel_params, upper_state_lifetime)
        dpdz = dPdZ(channel_params)
        convergence_checker = ConvergenceChecker(convergence_checking_interval, max_iterations, steady_state_tolerance,
                                                 stop_at_steady_state, ion_cross_section_areas[:n_ion_populations])
        P_out, N2_out, n_iter = self._bfecc_simulation(P, N2, dpdz, dn2dt, boundary_conditions, convergence_checker, dz,
                                                       dt, n_forward)
        P[:n_forward, :] = P_out[:n_forward, :-1]
        P[n_forward:, :] = P_out[n_forward:, 1:]
        N2[...] = N2_out[:, :-1]
        return n_iter

    def _bfecc_simulation(self, P_fiber_in, N2_in, dPdz, dN2dt,
                          boundary_conditions, convergence_checker, dz, dt, n_forward):
        P = np.zeros((P_fiber_in.shape[0], P_fiber_in.shape[1] + 1))
        P_hat_forward = np.empty_like(P)
        P_hat_backward = np.empty_like(P)
        P[:, :-1] = P_fiber_in
        P[:, -1] = P_fiber_in[:, -1]

        N2 = np.zeros((N2_in.shape[0], N2_in.shape[1] + 1))
        N2[:, :-1] = N2_in
        N2[:, -1] = N2_in[:, -1]

        idx_iteration = 0
        while convergence_checker.has_not_converged(N2, idx_iteration):
            boundary_conditions.apply_input(P, idx_iteration)
            boundary_conditions.apply_reflection(P)

            N2 += dt * dN2dt(P, N2)
            min_clamp(N2, SIMULATION_MIN_POWER)

            shift_to_propagation_direction_to_from(P_hat_forward, P, n_forward)
            boundary_conditions.apply_output(P_hat_forward, idx_iteration)
            P_hat_forward += dz * dPdz(P_hat_forward, N2)
            min_clamp(P_hat_forward, SIMULATION_MIN_POWER)

            shift_against_propagation_direction_to_from(P_hat_backward, P_hat_forward, n_forward)
            P_hat_backward -= dz * dPdz(P_hat_backward, N2)
            P = P_hat_forward + 0.5 * (P - P_hat_backward)
            min_clamp(P, SIMULATION_MIN_POWER)

            idx_iteration += 1

        boundary_conditions.apply_input(P, idx_iteration)
        boundary_conditions.apply_reflection(P)
        boundary_conditions.correct_output_by_reflection()
        return P, N2, idx_iteration
