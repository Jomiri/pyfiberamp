from scipy.integrate import solve_bvp
from functools import partial

from .simulation_result import SimulationResult
from .helper_funcs import *
from .initial_guess_maker import InitialGuessMaker




class FiberAmplifierSimulation:
    def __init__(self):
        self.fiber = None
        self.signal_gain_guess = 'linear'

        self.signal_wls = np.empty(0)
        self.signal_vs = np.empty(0)
        self.signal_dvs = np.empty(0)
        self.signal_us = np.empty(0)
        self.signal_powers = np.empty(0)
        self.signal_mode_field_radii = np.empty(0)
        self.signal_peak_power_func = lambda x: x

        self.co_pump_wls = np.empty(0)
        self.co_pump_vs = np.empty(0)
        self.co_pump_dvs = np.empty(0)
        self.co_pump_us = np.empty(0)
        self.co_pump_powers = np.empty(0)
        self.co_pump_mode_field_radii = np.empty(0)

        self.counter_pump_wls = np.empty(0)
        self.counter_pump_vs = np.empty(0)
        self.counter_pump_dvs = np.empty(0)
        self.counter_pump_us = np.empty(0)
        self.counter_pump_powers = np.empty(0)
        self.counter_pump_mode_field_radii = np.empty(0)

        self.n_ase_bins = 0
        self.ase_wls = np.empty(0)
        self.ase_vs = np.empty(0)
        self.forward_ase_vs = np.empty(0)
        self.backward_ase_vs = np.empty(0)
        self.ase_dvs = np.empty(0)
        self.forward_ase_dvs = np.empty(0)
        self.backward_ase_dvs = np.empty(0)
        self.backward_ase_us = np.empty(0)
        self.forward_ase_us = np.empty(0)
        self.ase_input_powers = np.empty(0)

        self.raman_is_included = False
        self.backward_raman_allowed = False
        self.raman_input_power = 0
        self.raman_wl_bandwidth = 0

        self.raman_vs = np.empty(0)
        self.raman_wls = np.empty(0)
        self.raman_effective_dvs = np.empty(0)
        self.forward_raman_dvs = np.empty(0)
        self.forward_raman_vs = np.empty(0)
        self.forward_raman_us = np.empty(0)
        self.backward_raman_dvs = np.empty(0)
        self.backward_raman_vs = np.empty(0)
        self.backward_raman_us = np.empty(0)
        self.raman_input_powers = np.empty(0)

        self.slices = {}

    @property
    def frequencies(self):
        """Frequencies (Hz) of the modeled optical signals."""
        vs = np.hstack((self.signal_vs,
                        self.co_pump_vs,
                        self.counter_pump_vs,
                        self.forward_ase_vs,
                        self.backward_ase_vs,
                        self.forward_raman_vs,
                        self.backward_raman_vs))
        return vs[:, np.newaxis]

    @property
    def frequency_bandwidths(self):
        """
        Frequency bandwidths of the modeled optical signals.
        Relevant to spontaneous emission and spontaneous Raman scattering.
        """
        dvs = np.hstack((self.signal_dvs,
                         self.co_pump_dvs,
                         self.counter_pump_dvs,
                         self.forward_ase_dvs,
                         self.backward_ase_dvs,
                         self.forward_raman_dvs,
                         self.backward_raman_dvs))
        return dvs[:, np.newaxis]

    @property
    def propagation_directions(self):
        """
        Directions of the modeled signals:
         +1 for forward propagation
         -1 for backward propagation
         """
        us = np.hstack((self.signal_us,
                        self.co_pump_us,
                        self.counter_pump_us,
                        self.forward_ase_us,
                        self.backward_ase_us,
                        self.forward_raman_us,
                        self.backward_raman_us))
        return us[:, np.newaxis]

    @property
    def number_of_modes(self):
        m = np.full(self.frequencies.shape, MODES_IN_SINGLE_MODE_FIBER)
        # Assuming PM Raman signal
        m[self.slices['forward_raman_slice']] = RAMAN_MODES_IN_PM_FIBER
        m[self.slices['backward_raman_slice']] = RAMAN_MODES_IN_PM_FIBER
        return m

    @property
    def predefined_mode_field_radii(self):
        mode_field_radii = np.zeros(len(self.frequencies))
        mode_field_radii[self.slices['signal_slice']] = np.array(self.signal_mode_field_radii)
        mode_field_radii[self.slices['co_pump_slice']] = np.array(self.co_pump_mode_field_radii)
        mode_field_radii[self.slices['counter_pump_slice']] = np.array(self.counter_pump_mode_field_radii)
        mode_field_radii.shape = self.frequencies.shape
        return mode_field_radii

    def add_cw_signal(self, wl, power, mode_field_radius=0.0):
        self.signal_wls = np.append(self.signal_wls, wl)
        self.signal_powers = np.append(self.signal_powers, power)
        self.signal_mode_field_radii = np.append(self.signal_mode_field_radii, mode_field_radius)

    def add_pulsed_signal(self, wl, power, f_rep, fwhm_duration, mode_field_radius=0.0):
        self.add_cw_signal(wl, power, mode_field_radius)
        self.signal_peak_power_func = partial(gaussian_peak_power, (f_rep, fwhm_duration))
        check_signal_reprate(f_rep, REP_RATE_LOWER_LIMIT)

    def add_co_pump(self, wl, power, mode_field_radius=0.0):
        self.co_pump_wls = np.append(self.co_pump_wls, wl)
        self.co_pump_powers = np.append(self.co_pump_powers, power)
        self.co_pump_mode_field_radii = np.append(self.counter_pump_mode_field_radii, mode_field_radius)

    def add_counter_pump(self, wl, power, mode_field_radius=0.0):
        self.counter_pump_wls = np.append(self.counter_pump_wls, wl)
        self.counter_pump_powers = np.append(self.counter_pump_powers, power)
        self.counter_pump_mode_field_radii = np.append(self.counter_pump_mode_field_radii, mode_field_radius)

    def add_ase(self, wl_start, wl_end, n_bins):
        assert (wl_end > wl_start)
        assert (n_bins > 0 and isinstance(n_bins, int))
        self.n_ase_bins = n_bins
        self.ase_wls = np.linspace(wl_start, wl_end, self.n_ase_bins)
        self.ase_vs = wl_to_freq(self.ase_wls)
        self.ase_input_powers = np.full(self.n_ase_bins, SIMULATION_MIN_POWER)
        ase_dv = 0 if self.n_ase_bins == 0 else (self.ase_vs[0] - self.ase_vs[-1]) / self.n_ase_bins
        self.ase_dvs = np.full(self.n_ase_bins, ase_dv)

    def add_raman(self, wl_bandwidth=RAMAN_GAIN_WL_BANDWIDTH, backward_raman_allowed=True, input_power=SIMULATION_MIN_POWER):
        self.raman_is_included = True
        self.raman_input_power = input_power
        self.raman_wl_bandwidth = wl_bandwidth
        self.backward_raman_allowed = backward_raman_allowed

    def reset_signals(self):
        self.signal_wls = np.empty(0)
        self.signal_powers = np.empty(0)
        self.signal_mode_field_radii = np.empty(0)
        self.signal_peak_power_func = None

    def run(self, npoints, tol=1e-3):
        self._initialize()

        boundary_condition_residual = self._make_boundary_condition_function()
        rate_equation_rhs, upper_level_func = self._make_rhs_and_upper_level_fraction_funcs()

        z = np.linspace(0, self.fiber.length, npoints)
        guess_maker = InitialGuessMaker(self._input_powers(), self.slices, z)
        guess = guess_maker.make_guess()
        sol = solve_bvp(rate_equation_rhs, boundary_condition_residual,
                        z, guess, max_nodes=SOLVER_MAX_NODES, tol=tol, verbose=2)
        return self._finalize(sol, upper_level_func)

    def _finalize(self, sol, upper_level_func):
        res = SimulationResult(sol)
        res.upper_level_fraction = upper_level_func(sol.y)
        res = self._add_wls_and_slices_to_result(res)
        return res

    def _initialize(self):
        self._init_raman()
        self._init_vs()
        self._init_dvs()
        self._init_us()
        self._init_slices()

    def _init_raman(self):
        if self.raman_is_included:
            self.raman_vs = wl_to_freq(self.signal_wls) - RAMAN_SHIFT_FREQ
            self.raman_wls = freq_to_wl(self.raman_vs)
            self.raman_input_powers = np.full(len(self.raman_vs), self.raman_input_power)
            raman_bandwidth = wl_bw_to_freq_bw(wl_bw=self.raman_wl_bandwidth, center_wl=self.raman_wls[0])
            self.raman_effective_dvs = np.full(len(self.raman_vs), raman_bandwidth)

    def _init_vs(self):
        self.signal_vs = wl_to_freq(self.signal_wls)
        self.co_pump_vs = wl_to_freq(self.co_pump_wls)
        self.counter_pump_vs = wl_to_freq(self.counter_pump_wls)
        self.forward_ase_vs = np.copy(self.ase_vs)
        self.backward_ase_vs = np.copy(self.ase_vs)
        self.forward_raman_vs = np.copy(self.raman_vs)
        self.backward_raman_vs = np.copy(self.raman_vs)

    def _init_dvs(self):
        self.signal_dvs = np.zeros(len(self.signal_wls))
        self.co_pump_dvs = np.zeros(len(self.co_pump_wls))
        self.counter_pump_dvs = np.zeros(len(self.counter_pump_wls))
        self.forward_ase_dvs = np.copy(self.ase_dvs)
        self.backward_ase_dvs = np.copy(self.ase_dvs)
        self.forward_raman_dvs = np.copy(self.raman_effective_dvs)
        self.backward_raman_dvs = np.copy(self.raman_effective_dvs)
        if not self.backward_raman_allowed:
            self.backward_raman_dvs = np.zeros_like(self.backward_raman_dvs)

    def _init_us(self):
        self.signal_us = np.ones(len(self.signal_vs))
        self.co_pump_us = np.ones(len(self.co_pump_vs))
        self.counter_pump_us = np.ones(len(self.counter_pump_vs)) * -1
        self.forward_ase_us = np.ones(len(self.ase_vs))
        self.backward_ase_us = np.ones(len(self.ase_vs)) * -1
        self.forward_raman_us = np.ones(len(self.raman_vs))
        self.backward_raman_us = np.ones(len(self.raman_vs)) * -1

    def _init_slices(self):
        n_signal = len(self.signal_wls)
        n_co_pump = len(self.co_pump_wls)
        n_counter_pump = len(self.counter_pump_wls)
        n_ase = self.n_ase_bins
        n_raman = len(self.raman_vs)

        co_pump_start = n_signal
        counter_pump_start = co_pump_start + n_co_pump
        forward_ase_start = counter_pump_start + n_counter_pump
        backward_ase_start = forward_ase_start + n_ase
        forward_raman_start = backward_ase_start + n_ase
        backward_raman_start = forward_raman_start + n_raman
        backward_raman_end = backward_raman_start + n_raman

        self.slices['signal_slice'] = slice(0, co_pump_start)
        self.slices['co_pump_slice'] = slice(co_pump_start, counter_pump_start)
        self.slices['counter_pump_slice'] = slice(counter_pump_start, forward_ase_start)
        self.slices['forward_ase_slice'] = slice(forward_ase_start, backward_ase_start)
        self.slices['backward_ase_slice'] = slice(backward_ase_start, forward_raman_start)
        self.slices['forward_raman_slice'] = slice(forward_raman_start, backward_raman_start)
        self.slices['backward_raman_slice'] = slice(backward_raman_start, backward_raman_end)

    def _make_boundary_condition_function(self):
        signal_slice = self.slices['signal_slice']
        co_pump_slice = self.slices['co_pump_slice']
        counter_pump_slice = self.slices['counter_pump_slice']
        forward_ase_slice = self.slices['forward_ase_slice']
        backward_ase_slice = self.slices['backward_ase_slice']
        forward_raman_slice = self.slices['forward_raman_slice']
        backward_raman_slice = self.slices['backward_raman_slice']

        expected = self._input_powers()

        def boundary_condition_func(powers_start, powers_end):
            current = np.hstack((powers_start[signal_slice],
                                 powers_start[co_pump_slice],
                                 powers_end[counter_pump_slice],
                                 powers_start[forward_ase_slice],
                                 powers_end[backward_ase_slice],
                                 powers_start[forward_raman_slice],
                                 powers_end[backward_raman_slice]))
            return current - expected

        return boundary_condition_func

    def _input_powers(self):
        return np.hstack((self.signal_powers,
                          self.co_pump_powers,
                          self.counter_pump_powers,
                          self.ase_input_powers,
                          self.ase_input_powers,
                          self.raman_input_powers,
                          self.raman_input_powers))

    def _make_rhs_and_upper_level_fraction_funcs(self):
        v = self.frequencies
        dv = self.frequency_bandwidths
        u = self.propagation_directions
        m = self.number_of_modes
        mfr = self.predefined_mode_field_radii
        a = self.fiber.make_absorption_spectrum(v, mfr, self.slices)
        g = self.fiber.make_gain_spectrum(v, mfr, self.slices)
        eta = self.fiber.eta()
        h_v_eta = h * v * eta
        h_v_dv = h * v * dv
        g_m_h_v_dv = g * m * h_v_dv
        a_g = a + g
        a_term = a / h_v_eta
        ag_term = a_g / h_v_eta
        loss = self.fiber.background_loss
        a_l = a + loss

        def upper_level_excitation(P):
            return np.sum(P * a_term, axis=0) / (1 + np.sum(P * ag_term, axis=0))

        def rhs(_, P):
            n2_per_nt_normal = upper_level_excitation(P)
            return u * ((a_g * n2_per_nt_normal - a_l) * P + n2_per_nt_normal * g_m_h_v_dv)

        if self.raman_is_included:
            signal_slice = self.slices['signal_slice']
            forward_raman_slice = self.slices['forward_raman_slice']
            backward_raman_slice = self.slices['backward_raman_slice']
            h_v_dv_raman = h_v_dv[forward_raman_slice]
            g_r_forward = G_RAMAN
            g_r_backward = G_RAMAN if self.backward_raman_allowed else 0
            photon_energy_ratio = self.signal_vs / self.raman_vs
            area_eff = self.fiber.nonlinear_effective_area(self.signal_vs)

            def rhs(_, P):
                n2_per_nt_normal = upper_level_excitation(P)
                dPdz = u * ((a_g * n2_per_nt_normal - a_l) * P + n2_per_nt_normal * g_m_h_v_dv)
                signal_intensities = self.signal_peak_power_func(P[signal_slice, :]) / area_eff
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
        res.signal_wls = self.signal_wls
        res.co_pump_wls = self.co_pump_wls
        res.counter_pump_wls = self.counter_pump_wls
        res.ase_wls = self.ase_wls
        res.raman_wls = self.raman_wls
        return res




