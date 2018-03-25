from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from fiberamp.helper_funcs import *
from .fiber_base import FiberBase

CROSS_SECTION_SMOOTHING_FACTOR = 1e-51
SPECTRUM_PLOT_NPOINTS = 1000


class ActiveFiber(FiberBase):
    def __init__(self, length, absorption_cs_file, gain_cs_file,
                 core_r, upper_state_lifetime, ion_number_density,
                 background_loss, core_na):
        super().__init__(length, core_r, background_loss, core_na)

        absorption_spectrum = load_two_column_file(absorption_cs_file)
        gain_spectrum = load_two_column_file(gain_cs_file)

        absorption_spectrum[:, 0] *= 1e-9
        gain_spectrum[:, 0] *= 1e-9
        self.absorption_spectrum = absorption_spectrum
        self.gain_spectrum = gain_spectrum
        self.absorption_cs_interp = self.make_interpolate(absorption_spectrum)
        self.gain_cs_interp = self.make_interpolate(gain_spectrum)

        self.tau = upper_state_lifetime
        self.ion_number_density = ion_number_density
        self.overlap = None

    @staticmethod
    def make_interpolate(spectrum):
        frequency = wl_to_freq(spectrum[::-1, 0])
        cross_section = spectrum[::-1, 1]
        spline = UnivariateSpline(frequency, cross_section, s=CROSS_SECTION_SMOOTHING_FACTOR, ext='zeros')

        def interp(freq):
            cross_sec = spline(freq)
            cross_sec[cross_sec < 0] = 0
            return cross_sec

        return interp

    @property
    def overlap_is_constant(self):
        return self.overlap is not None

    def use_constant_overlap(self, overlap):
        self.overlap = overlap

    def use_automatic_overlap(self):
        self.overlap = None

    def make_single_overlap(self, mode_field_radius):
        if self.overlap_is_constant:
            return self.overlap
        return overlap_integral(self.core_r, mode_field_radius)

    def create_forward_pump_channel(self, wl, power, preset_mfd):
        return self.create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=+1)

    def create_backward_pump_channel(self, wl, power, preset_mfd):
        return self.create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=-1)
    def get_channel_gain(self, freq, mode_field_radius):

        return self.gain_cs_interp(freq) * self.make_single_overlap(mode_field_radius) * self.ion_number_density

    def get_channel_absorption(self, freq, mode_field_radius):
        return self.absorption_cs_interp(freq) * self.make_single_overlap(mode_field_radius) * self.ion_number_density

    def saturation_parameter(self):
        return eta_from_fiber_parameters(self.core_r, self.tau, self.ion_number_density)

    def finite_bandwidth_gain(self, center_frequency, frequency_bandwidth, mode_field_radius):
        start_frequency = center_frequency - frequency_bandwidth / 2
        end_frequency = center_frequency + frequency_bandwidth / 2
        start_gain = self.get_channel_gain(start_frequency, mode_field_radius)
        middle_gain = self.get_channel_gain(center_frequency, mode_field_radius)
        end_gain = self.get_channel_gain(end_frequency, mode_field_radius)
        return np.mean([start_gain, middle_gain, end_gain])

    def finite_bandwidth_absorption(self, center_frequency, frequency_bandwidth, mode_field_radius):
        start_frequency = center_frequency - frequency_bandwidth / 2
        end_frequency = center_frequency + frequency_bandwidth / 2
        start_absorption = self.get_channel_absorption(start_frequency, mode_field_radius)
        middle_absorption = self.get_channel_absorption(center_frequency, mode_field_radius)
        end_absorption = self.get_channel_absorption(end_frequency, mode_field_radius)
        return np.mean([start_absorption, middle_absorption, end_absorption])

    def plot_gain_and_absorption_spectrum(self):
        fig, ax = plt.subplots()
        gain = self.gain_spectrum
        absorption = self.absorption_spectrum
        gain_wls = np.linspace(gain[0, 0], gain[-1, 0], SPECTRUM_PLOT_NPOINTS)
        gain_vs = wl_to_freq(gain_wls)
        absorption_wls = np.linspace(absorption[0, 0], absorption[-1, 0], SPECTRUM_PLOT_NPOINTS)
        absorption_vs = wl_to_freq(absorption_wls)
        ax.plot(gain[:, 0] * 1e9, gain[:, 1], label='Gain')
        ax.plot(absorption[:, 0] * 1e9, absorption[:, 1], label='Absorption')
        ax.plot(absorption_wls * 1e9, self.absorption_cs_interp(absorption_vs), label='Absorption spline')
        ax.plot(gain_wls * 1e9, self.gain_cs_interp(gain_vs), label='Gain spline')
        ax.legend()
        ax.set_xlabel('Wavelength (nm)', fontsize=18)
        ax.set_ylabel('Gain/Absorption cross sections', fontsize=18)
