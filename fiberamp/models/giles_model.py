from fiberamp.helper_funcs import *


class GilesModel:
    def __init__(self, channels, fiber, *extra):
        self.v = channels.get_frequencies()[:, np.newaxis]
        self.dv = channels.get_frequency_bandwidths()[:, np.newaxis]
        self.u = channels.get_propagation_directions()[:, np.newaxis]
        self.m = channels.get_number_of_modes()[:, np.newaxis]
        self.a = channels.get_absorption()[:, np.newaxis]
        self.g = channels.get_gain()[:, np.newaxis]
        self.loss = channels.get_background_loss()[:, np.newaxis]
        self.fiber = fiber

    def make_rate_equation_rhs(self):
        eta = self.fiber.saturation_parameter()
        h_v_eta = h * self.v * eta
        h_v_dv = h * self.v * self.dv
        g_m_h_v_dv = self.g * self.m * h_v_dv
        a_g = self.a + self.g
        a_term = self.a / h_v_eta
        ag_term = a_g / h_v_eta
        a_l = self.a + self.loss
        u = self.u

        def upper_level_excitation(P):
            return np.sum(P * a_term, axis=0) / (1 + np.sum(P * ag_term, axis=0))

        def rhs(_, P):
            n2_per_nt_normal = np.sum(P * a_term, axis=0) / (1 + np.sum(P * ag_term, axis=0))
            return u * ((a_g * n2_per_nt_normal - a_l) * P + n2_per_nt_normal * g_m_h_v_dv)

        def amplifier_rate_equation(z, P):
            P[P < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER
            return rhs(z, P)

        return amplifier_rate_equation, upper_level_excitation



