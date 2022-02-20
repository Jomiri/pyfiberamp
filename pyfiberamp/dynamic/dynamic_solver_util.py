import numpy as np
from pyfiberamp.helper_funcs import *


def reorganize_by_ion_population(arr, num_ion_populations, n_channels):
    out = np.empty_like(arr)
    for i in range(num_ion_populations):
        out[i * n_channels:(i + 1) * n_channels, :] = arr[i::num_ion_populations, :]
    return out


class dNdT:
    def __init__(self, channel_params, tau):
        self.A = 1 / tau
        a_g_per_Nt = (channel_params.a + channel_params.g) / channel_params.ion_number_densities
        h_v_pi_r2 = h * channel_params.v * channel_params.ion_cross_section_areas
        self.a_per_h_v_pi_r2 = channel_params.a / h_v_pi_r2
        self.a_g_per_h_v_pi_r2_Nt = a_g_per_Nt / h_v_pi_r2

        self.n_channels = channel_params.n_channels
        self.num_ion_populations = channel_params.ion_number_densities.shape[0] // self.n_channels

        self.a_g_per_h_v_pi_r2_Nt = reorganize_by_ion_population(self.a_g_per_h_v_pi_r2_Nt,
                                                                 self.num_ion_populations, self.n_channels)
        self.a_per_h_v_pi_r2 = reorganize_by_ion_population(self.a_per_h_v_pi_r2,
                                                            self.num_ion_populations, self.n_channels)

    def __call__(self, P, N2):
        out = np.empty_like(N2)
        for i in range(self.num_ion_populations):
            start = i * self.n_channels
            end = start + self.n_channels
            out[i, :] = np.sum(P * (self.a_per_h_v_pi_r2[start:end, :]
                                    - self.a_g_per_h_v_pi_r2_Nt[start:end, :] * N2[i, :]), axis=0) - self.A * N2[i, :]
        return out


class dPdZ:
    def __init__(self, channel_params):
        self.a_g_per_Nt = (channel_params.a + channel_params.g) / channel_params.ion_number_densities
        self.a_l = channel_params.a + channel_params.l
        m = NUMBER_OF_ASE_POLARIZATION_MODES
        self.g_m_h_v_dv_per_Nt = m * h * channel_params.g * channel_params.v * channel_params.dv / channel_params.ion_number_densities
        self.n_channels = channel_params.n_channels
        self.num_ion_populations = channel_params.ion_number_densities.shape[0] // self.n_channels
        self.a_g_per_Nt = reorganize_by_ion_population(self.a_g_per_Nt, self.num_ion_populations, self.n_channels)
        self.a_l = reorganize_by_ion_population(self.a_l, self.num_ion_populations, self.n_channels)
        self.g_m_h_v_dv_per_Nt = reorganize_by_ion_population(self.g_m_h_v_dv_per_Nt, self.num_ion_populations, self.n_channels)

    def __call__(self, P, N2):
        out = np.zeros_like(P)
        for i in range(self.num_ion_populations):
            start = i * self.n_channels
            end = start + self.n_channels
            out[...] += P * (self.a_g_per_Nt[start:end, :] * N2[i, :] - self.a_l[start:end, :]) + self.g_m_h_v_dv_per_Nt[start:end, :] * N2[i, :]
        return out


class ChannelParameters:
    def __init__(self, a, g, l, v, dv, nodes, ion_cross_section_areas, ion_number_densities, n_channels):
        simulation_array_shape = (len(a), nodes + 1) # Extra boundary point
        self.a = a[:, np.newaxis] * np.ones(simulation_array_shape)
        self.g = g[:, np.newaxis] * np.ones(simulation_array_shape)
        self.l = l[:, np.newaxis] * np.ones(simulation_array_shape)
        self.v = v[:, np.newaxis] * np.ones(simulation_array_shape)
        self.dv = dv[:, np.newaxis] * np.ones(simulation_array_shape)
        self.ion_cross_section_areas = ion_cross_section_areas[:, np.newaxis] * np.ones(simulation_array_shape)
        self.ion_number_densities = ion_number_densities[:, np.newaxis] * np.ones(simulation_array_shape)
        self.n_channels = n_channels

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