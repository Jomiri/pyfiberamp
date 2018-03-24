from scipy.integrate import solve_bvp
from functools import partial

from .simulation_result import SimulationResult
from .helper_funcs import *
from .initial_guess_maker import InitialGuessMaker
from .optical_channel import SimulationChannels


class FiberAmplifierSimulation:
    def __init__(self, fiber):
        self.fiber = fiber
        self.channels = SimulationChannels(fiber)
        self.backward_raman_allowed = False
        self.raman_is_included = False
        self.slices = {}

    def add_cw_signal(self, wl, power, mode_field_diameter=0.0):
        self.channels.add_forward_signal(wl, power, mode_field_diameter)

    def add_pulsed_signal(self, wl, power, f_rep, fwhm_duration, mode_field_diameter=0.0):
        self.channels.add_pulsed_forward_signal(wl, power, f_rep, fwhm_duration, mode_field_diameter)

    def add_co_pump(self, wl, power, mode_field_diameter=0.0):
        self.channels.add_forward_pump(wl, power, mode_field_diameter)

    def add_counter_pump(self, wl, power, mode_field_diameter=0.0):
        self.channels.add_backward_pump(wl, power, mode_field_diameter)

    def add_ase(self, wl_start, wl_end, n_bins):
        self.channels.add_ase(wl_start, wl_end, n_bins)

    def add_raman(self, backward_raman_allowed=True, input_power=SIMULATION_MIN_POWER):
        self.channels.add_raman(input_power, backward_raman_allowed)
        self.raman_is_included = True
        self.backward_raman_allowed = backward_raman_allowed

    def run(self, npoints, tol=1e-3):
        self._init_slices()

        boundary_condition_residual = self._make_boundary_condition_function()
        rate_equation_rhs, upper_level_func = self._make_rhs_and_upper_level_fraction_funcs()

        z = np.linspace(0, self.fiber.length, npoints)
        guess_maker = InitialGuessMaker(self.channels.get_input_powers(), self.slices, z)
        guess = guess_maker.make_guess()
        sol = solve_bvp(rate_equation_rhs, boundary_condition_residual,
                        z, guess, max_nodes=SOLVER_MAX_NODES, tol=tol, verbose=2)
        return self._finalize(sol, upper_level_func)

    def _finalize(self, sol, upper_level_func):
        res = SimulationResult(sol)
        res.upper_level_fraction = upper_level_func(sol.y)
        res = self._add_wls_and_slices_to_result(res)
        return res

    def _init_slices(self):
        self.slices = self.channels.get_slices()

    def _make_boundary_condition_function(self):
        forward_signal_slice = self.slices['forward_signal_slice']
        backward_signal_slice = self.slices['backward_signal_slice']
        forward_pump_slice = self.slices['forward_pump_slice']
        backward_pump_slice = self.slices['backward_pump_slice']
        forward_ase_slice = self.slices['forward_ase_slice']
        backward_ase_slice = self.slices['backward_ase_slice']
        forward_raman_slice = self.slices['forward_raman_slice']
        backward_raman_slice = self.slices['backward_raman_slice']

        expected = self.channels.get_input_powers()

        def boundary_condition_func(powers_start, powers_end):
            current = np.hstack((powers_start[forward_signal_slice],
                                 powers_end[backward_signal_slice],
                                 powers_start[forward_pump_slice],
                                 powers_end[backward_pump_slice],
                                 powers_start[forward_ase_slice],
                                 powers_end[backward_ase_slice],
                                 powers_start[forward_raman_slice],
                                 powers_end[backward_raman_slice]))
            return current - expected

        return boundary_condition_func

    def _make_rhs_and_upper_level_fraction_funcs(self):
        v = self.channels.get_frequencies()[:, np.newaxis]
        dv = self.channels.get_frequency_bandwidths()[:, np.newaxis]
        u = self.channels.get_propagation_directions()[:, np.newaxis]
        m = self.channels.get_number_of_modes()[:, np.newaxis]
        a = self.channels.get_absorption()[:, np.newaxis]
        g = self.channels.get_gain()[:, np.newaxis]
        loss = self.channels.get_background_loss()[:, np.newaxis]
        eta = self.fiber.eta()
        h_v_eta = h * v * eta
        h_v_dv = h * v * dv
        g_m_h_v_dv = g * m * h_v_dv
        a_g = a + g
        a_term = a / h_v_eta
        ag_term = a_g / h_v_eta
        a_l = a + loss

        def upper_level_excitation(P):
            return np.sum(P * a_term, axis=0) / (1 + np.sum(P * ag_term, axis=0))

        def rhs(_, P):
            n2_per_nt_normal = upper_level_excitation(P)
            return u * ((a_g * n2_per_nt_normal - a_l) * P + n2_per_nt_normal * g_m_h_v_dv)

        if self.raman_is_included:
            signal_slice = self.slices['forward_signal_slice']
            forward_raman_slice = self.slices['forward_raman_slice']
            signal_vs = v[signal_slice]
            raman_vs = v[forward_raman_slice]
            backward_raman_slice = self.slices['backward_raman_slice']
            h_v_dv_raman = h_v_dv[forward_raman_slice]
            g_r_forward = G_RAMAN
            g_r_backward = G_RAMAN if self.backward_raman_allowed else 0
            photon_energy_ratio = signal_vs / raman_vs
            area_eff = self.fiber.nonlinear_effective_area(signal_vs)
            signal_peak_power_func = self.channels.forward_signals[0].peak_power_func

            def rhs(_, P):
                n2_per_nt_normal = upper_level_excitation(P)
                dPdz = u * ((a_g * n2_per_nt_normal - a_l) * P + n2_per_nt_normal * g_m_h_v_dv)
                signal_intensities = signal_peak_power_func(P[signal_slice, :]) / area_eff
                forward_raman_growth = g_r_forward * signal_intensities * (h_v_dv_raman + P[forward_raman_slice, :])
                backward_raman_growth = g_r_backward * signal_intensities * (h_v_dv_raman + P[backward_raman_slice, :])
                dPdz[forward_raman_slice, :] += forward_raman_growth
                dPdz[backward_raman_slice, :] -= backward_raman_growth
                dPdz[signal_slice, :] -= photon_energy_ratio * (forward_raman_growth + backward_raman_growth)
                return dPdz

        def amplifier_rate_equation(z, P):
            P[P < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER
            return rhs(z, P)

        return amplifier_rate_equation, upper_level_excitation

    def _add_wls_and_slices_to_result(self, res):
        res.slices = self.slices
        res.backward_raman_allowed = self.backward_raman_allowed
        res.wavelengths = self.channels.get_wavelengths()
        return res




