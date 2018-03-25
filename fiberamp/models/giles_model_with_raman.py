from .giles_model import GilesModel
from ..helper_funcs import *


class GilesModelWithRaman(GilesModel):
    def __init__(self, channels, fiber, *extra):
        super().__init__(channels, fiber, *extra)
        self.slices = channels.get_slices()
        self.peak_power_func = channels.forward_signals[0].peak_power_func
        self.backward_raman_allowed = channels.backward_raman_allowed

    def make_rate_equation_rhs(self):
        h_v_eta = h * self.v * self.fiber.saturation_parameter()
        h_v_dv = h * self.v * self.dv
        g_m_h_v_dv = self.g * self.m * h_v_dv
        a_g = self.a + self.g
        a_term = self.a / h_v_eta
        ag_term = a_g / h_v_eta
        a_l = self.a + self.loss
        u = self.u

        signal_slice = self.slices['forward_signal_slice']
        forward_raman_slice = self.slices['forward_raman_slice']
        backward_raman_slice = self.slices['backward_raman_slice']
        signal_vs = self.v[signal_slice]
        raman_vs = self.v[forward_raman_slice]
        h_v_dv_raman = h_v_dv[forward_raman_slice]
        g_r_forward = G_RAMAN
        g_r_backward = G_RAMAN if self.backward_raman_allowed else 0
        photon_energy_ratio = signal_vs / raman_vs
        effective_area = self.fiber.nonlinear_effective_area(signal_vs)
        signal_peak_power_func = self.peak_power_func

        def upper_level_excitation(P):
            return np.sum(P * a_term, axis=0) / (1 + np.sum(P * ag_term, axis=0))

        def rhs(_, P):
            n2_per_nt_normal = upper_level_excitation(P)
            dPdz = u * ((a_g * n2_per_nt_normal - a_l) * P + n2_per_nt_normal * g_m_h_v_dv)
            signal_intensities = signal_peak_power_func(P[signal_slice, :]) / effective_area
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