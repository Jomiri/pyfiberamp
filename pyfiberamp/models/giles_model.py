from pyfiberamp.helper_funcs import *


class GilesModel:
    """GilesModel defines the rate equations of the steady state Giles model. See
     C.R. Giles and E. Desurvire, "Modeling erbium-doped fiber amplifiers," in Journal of Lightwave Technology, vol. 9,
     no. 2, pp. 271-283, Feb 1991. doi: 10.1109/50.65886
     """
    def __init__(self, channels, fiber):
        """The model is initialized with the optical channels and the simulated fiber. The parameters are named in the
        same way as in the paper by Giles and Desurvire."""
        self.v = np.array(channels.get_frequencies())[:, np.newaxis]
        self.dv = np.array(channels.get_frequency_bandwidths())[:, np.newaxis]
        self.u = np.array(channels.get_propagation_directions())[:, np.newaxis]
        self.m = np.array(channels.get_number_of_modes())[:, np.newaxis]
        self.a = np.array(channels.get_absorption())[:, np.newaxis]
        self.g = np.array(channels.get_gain())[:, np.newaxis]
        self.loss = np.array(channels.get_background_loss())[:, np.newaxis]
        self.fiber = fiber

    def make_rate_equation_rhs(self):
        # First, we precalculate all the constants
        zeta = self.fiber.saturation_parameter()
        h_v_zeta = h * self.v * zeta
        h_v_dv = h * self.v * self.dv
        g_m_h_v_dv = self.g * self.m * h_v_dv
        a_g = self.a + self.g
        a_per_h_v_zeta = self.a / h_v_zeta
        a_g_per_h_v_zeta = a_g / h_v_zeta
        a_l = self.a + self.loss
        u = self.u

        def upper_level_excitation(P):
            """The fraction of ions in the upper level (n_2 / n_t in the Giles model)"""
            return np.sum(P * a_per_h_v_zeta, axis=0) / (1 + np.sum(P * a_g_per_h_v_zeta, axis=0))

        def rhs(_, P):
            """The rate equation for the optical powers (dP/dz in the Giles model)"""
            n2_per_nt = upper_level_excitation(P)
            return u * ((a_g * n2_per_nt - a_l) * P + n2_per_nt * g_m_h_v_dv)

        def amplifier_rate_equation(z, P):
            """The rate equation function that is passed to the solver. The "hack" limits all powers to small positive
            values to prevent the solver from moving to negative values and diverging."""
            P[P < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER
            return rhs(z, P)

        return amplifier_rate_equation, upper_level_excitation



