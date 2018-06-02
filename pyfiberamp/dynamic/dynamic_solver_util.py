import numpy as np
from pyfiberamp.helper_funcs import *


class dNdT:
    def __init__(self, channel_params, core_radius, ion_number_density, tau):
        self.A = 1 / tau
        a_g_per_Nt = (channel_params.a + channel_params.g) / ion_number_density
        h_v_pi_r2 = h * channel_params.v * np.pi * core_radius**2
        self.a_per_h_v_pi_r2 = channel_params.a / h_v_pi_r2
        self.a_g_per_h_v_pi_r2_Nt = a_g_per_Nt / h_v_pi_r2

    def __call__(self, P, N2):
        return np.sum(P * (self.a_per_h_v_pi_r2 - self.a_g_per_h_v_pi_r2_Nt * N2), axis=0) - self.A * N2


class dPdZ:
    def __init__(self, channel_params, ion_number_density):
        self.a_g_per_Nt = (channel_params.a + channel_params.g) / ion_number_density
        self.a_l = channel_params.a + channel_params.l
        m = NUMBER_OF_MODES_IN_SINGLE_MODE_FIBER
        self.g_m_h_v_dv_per_Nt = m * h * channel_params.g * channel_params.v * channel_params.dv / ion_number_density

    def __call__(self, P, N2):
        return P * (self.a_g_per_Nt * N2 - self.a_l) + self.g_m_h_v_dv_per_Nt * N2


class ChannelParameters:
    def __init__(self, a, g, l, v, dv, nodes):
        simulation_array_shape = (len(a), nodes + 1) # Extra boundary point
        self.a = a[:, np.newaxis] * np.ones(simulation_array_shape)
        self.g = g[:, np.newaxis] * np.ones(simulation_array_shape)
        self.l = l[:, np.newaxis] * np.ones(simulation_array_shape)
        self.v = v[:, np.newaxis] * np.ones(simulation_array_shape)
        self.dv = dv[:, np.newaxis] * np.ones(simulation_array_shape)

        # Boundary points don't have ions
        self.a[:, 0] = 0
        self.a[:, -1] = 0
        self.g[:, 0] = 0
        self.g[:, -1] = 0
        self.l[:, 0] = 0
        self.l[:, -1] = 0


def shift_against_propagation_direction_to_from(dest_arr, source_arr, n_forward):
    dest_arr[:n_forward, :-1] = source_arr[:n_forward, 1:]
    dest_arr[n_forward:, 1:] = source_arr[n_forward:, :-1]


def shift_to_propagation_direction_to_from(dest_arr, source_arr, n_forward):
    dest_arr[:n_forward, 1:] = source_arr[:n_forward, :-1]
    dest_arr[n_forward:, :-1] = source_arr[n_forward:, 1:]