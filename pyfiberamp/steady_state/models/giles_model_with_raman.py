from .giles_model import GilesModel
from pyfiberamp.helper_funcs import *


class GilesModelWithRaman(GilesModel):
    """GilesModelWithRaman extends the Giles model with a simple rate equation description of Raman scattering. See
    R.G. Smith, "Optical Power Handling Capacity of Low Loss Optical Fibers as Determined by Stimulated Raman and
    Brillouin Scattering," Appl. Opt. 11, 2489-2494 (1972)
    """
    raman_gain = DEFAULT_RAMAN_GAIN

    def __init__(self, channels, fiber):
        """The model is initialized with the optical channels and the simulated fiber. The parameters are named in the
        same way as in the paper by Giles and Desurvire. A pulsed signal is taken into account in the peak input_power
        function."""
        super().__init__(channels, fiber)
        self.peak_power_func = channels.forward_signals[0].peak_power_func
        if fiber.effective_area_type == 'mode':
            self.signal_effective_area = channels.forward_signals[0].mode.nonlinear_effective_area()
        elif fiber.effective_area_type == 'core':
            self.signal_effective_area = fiber.core_area()
        self.backward_raman_allowed = channels.backward_raman_allowed
        self.slices = channels.get_slices()

    def make_rate_equation_rhs(self):
        # Precalculated constants related to the classical Giles model
        h_v_zeta = h * self.v * self.fiber.saturation_parameter()
        h_v_dv = h * self.v * self.dv
        spontaneous_emission_seeding = self.g * self.m * h_v_dv
        a_g = self.a + self.g
        a_per_h_v_zeta = self.a / h_v_zeta
        a_g_per_h_v_zeta = a_g / h_v_zeta
        a_l = self.a + self.loss
        u = self.u

        # Precalculated constants related to Raman scattering
        slices = self.slices
        signal_slice = slices['forward_signal']
        forward_raman_slice = slices['forward_raman']
        backward_raman_slice = slices['backward_raman']
        signal_vs = self.v[signal_slice]
        raman_vs = self.v[forward_raman_slice]
        h_v_dv_forward_raman = h_v_dv[forward_raman_slice]
        h_v_dv_backward_raman = h_v_dv[backward_raman_slice]
        g_r_forward = self.raman_gain
        g_r_backward = self.raman_gain if self.backward_raman_allowed else 0
        photon_energy_ratio = signal_vs / raman_vs

        def signal_intensity(P_signal):
            """Calculates the intensity of the signal using the peak input_power if the signal is pulsed and the nonlinear
            effective area of the fiber."""
            peak_power = self.peak_power_func(P_signal)
            return peak_power / self.signal_effective_area

        def upper_level_excitation(P):
            """The fraction of ions in the upper level (n_2 / n_t in the Giles model)"""
            return np.sum(P * a_per_h_v_zeta, axis=0) / (1 + np.sum(P * a_g_per_h_v_zeta, axis=0))

        def rhs(_, P):
            """The combined rate equation of the Giles model and the Raman model."""
            n2_per_nt = upper_level_excitation(P)
            I_signal = signal_intensity(P[signal_slice, :])

            forward_raman_growth = g_r_forward * I_signal * (h_v_dv_forward_raman + P[forward_raman_slice, :])
            backward_raman_growth = g_r_backward * I_signal * (h_v_dv_backward_raman + P[backward_raman_slice, :])
            signal_depletion = photon_energy_ratio * (forward_raman_growth + backward_raman_growth)

            dPdz = u * ((a_g * n2_per_nt - a_l) * P + n2_per_nt * spontaneous_emission_seeding)
            dPdz[forward_raman_slice, :] += forward_raman_growth
            dPdz[backward_raman_slice, :] -= backward_raman_growth
            dPdz[signal_slice, :] -= signal_depletion
            return dPdz

        def amplifier_rate_equation(z, P):
            """The rate equation function that is passed to the solver. The "hack" limits all powers to small positive
            values to prevent the solver from moving to negative values and diverging."""
            P[P < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER
            return rhs(z, P)

        return amplifier_rate_equation, upper_level_excitation
