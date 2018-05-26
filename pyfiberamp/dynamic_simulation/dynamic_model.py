
import numpy as np
from pyfiberamp.helper_funcs import *
from pyfiberamp.util import SlicedArray


class DynamicModel:

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
        self.channels = channels

    def fiber_response_functions(self):
        # First, we precalculate all the constants
        tau = self.fiber.spectroscopy.upper_state_lifetime
        h_v_dv = h * self.v * self.dv
        g_m_h_v_dv = self.g * self.m * h_v_dv
        a_g = self.a + self.g
        a_l = self.a + self.loss
        a = self.a
        u = self.u
        Nt = self.fiber.ion_number_density
        A = 1 / tau
        h_v_pi_r2_inv = 1 / (h * self.v * np.pi * self.fiber.core_radius**2)

        def F(P, n2):
            gain = a_g * n2/Nt - a_l
            return u * (gain * P + g_m_h_v_dv * n2 / Nt)

        def dN2dt(P, n2):
            return np.sum(P * h_v_pi_r2_inv * (a - a_g * n2 / Nt), axis=0) - n2 * A

        return F, dN2dt


