from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from pyfiberamp.helper_funcs import *
from .fiber_base import FiberBase


class ActiveFiber(FiberBase):
    """ActiveFiber describes a step-index single-mode fiber with active dopant ions. Currently, only uniform doping
    in the whole core area is supported. This class extends the FiberBase class by adding spectroscopic data: gain and
    emission spectra, upper state lifetime and doping concentration."""
    def __init__(self, length=0, absorption_cs_file=None, gain_cs_file=None,
                 core_radius=0, upper_state_lifetime=0, ion_number_density=0,
                 background_loss=0, core_na=0):
        """

        :param length: Fiber length
        :type length: float
        :param absorption_cs_file: Name of the file containing absorption cross-section data
        :type absorption_cs_file: str
        :param gain_cs_file: Name of the file containing emission cross-section data
        :type gain_cs_file: str
        :param core_radius: Core radius
        :type core_radius: float
        :param upper_state_lifetime: Lifetime of the excited state
        :type upper_state_lifetime: float
        :param ion_number_density: Number density of the dopant ions (1/m^3)
        :type ion_number_density: float
        :param background_loss: Linear loss of the core (1/m, NOT in dB/m)
        :type background_loss: float
        :param core_na: Numerical aperture of the core
        :type core_na: float

        """
        super().__init__(length=length,
                         core_radius=core_radius,
                         background_loss=background_loss,
                         core_na=core_na)

        self.absorption_spectrum = load_spectrum(absorption_cs_file)
        self.gain_spectrum = load_spectrum(gain_cs_file)

        self.absorption_cs_interp = self._make_cross_section_interpolate(self.absorption_spectrum)
        self.gain_cs_interp = self._make_cross_section_interpolate(self.gain_spectrum)

        self.tau = upper_state_lifetime
        self.ion_number_density = ion_number_density
        self.overlap = None

    @staticmethod
    def _make_cross_section_interpolate(spectrum):
        """Creates a cubic spline interpolate from the imported cross section data. Cross section is assumed to be
        zero outside the imported data range."""
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
        """Returns True if the user has set a default overlap value. Overlap is the Giles model overlap between the mode
        and the dopant ions (= core in this case)."""
        return self.overlap is not None

    def use_constant_overlap(self, overlap):
        """Setter for a user-provided overlap value.

        :param overlap: Single overlap value that will be used for all optical channels.
        :type overlap: float

        """
        self.overlap = overlap

    def use_automatic_overlap(self):
        """Disables the overlap value given by the user."""
        self.overlap = None

    def _make_single_overlap(self, mode_field_radius):
        """Returns the overlap value for a given mode field radius."""
        if self.overlap_is_constant:
            return self.overlap
        return overlap_integral(self.core_radius, mode_field_radius)

    def create_forward_pump_channel(self, wl, power, preset_mfd):
        """Wrapper method for "create_in_core_single_frequency_channel" to hide the direction parameter."""
        return self._create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=+1)

    def create_backward_pump_channel(self, wl, power, preset_mfd):
        """Wrapper method for "create_in_core_single_frequency_channel" to hide the direction parameter."""
        return self._create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=-1)

    def _get_channel_gain(self, freq, mode_field_radius):
        """This is the maximum gain g* defined in the Giles model. The gain for a mode with a given frequency depends
        on the the emission cross section, overlap between mode and core/ions and the doping concentration."""
        return self.gain_cs_interp(freq) * self._make_single_overlap(mode_field_radius) * self.ion_number_density

    def _get_channel_absorption(self, freq, mode_field_radius):
        """This is the maximum absorption alpha defined in the Giles model. The absorption for a mode with a given
         frequency depends on the the absorption cross section, overlap between mode and core/ions and the
         doping concentration."""
        return self.absorption_cs_interp(freq) * self._make_single_overlap(mode_field_radius) * self.ion_number_density

    def saturation_parameter(self):
        """Returns the constant saturation parameter zeta defined in the Giles model."""
        return zeta_from_fiber_parameters(self.core_radius, self.tau, self.ion_number_density)

    def _finite_bandwidth_gain(self, center_frequency, frequency_bandwidth, mode_field_radius):
        """Calculates the maximum gain g* for a finite bandwidth signal by averaging over the start, end and middle
        points."""
        return self._averaged_value_of_finite_bandwidth_spectrum(center_frequency, frequency_bandwidth,
                                                                 mode_field_radius, self._get_channel_gain)

    def _finite_bandwidth_absorption(self, center_frequency, frequency_bandwidth, mode_field_radius):
        """Calculates the maximum absorption alpha for a finite bandwidth signal by averaging over the start, end
        and middle points."""
        return self._averaged_value_of_finite_bandwidth_spectrum(center_frequency, frequency_bandwidth,
                                                                 mode_field_radius, self._get_channel_absorption)

    def _averaged_value_of_finite_bandwidth_spectrum(self, center_frequency, frequency_bandwidth, mode_field_radius,
                                                     spectrum_func):
        """Actual function used to calculate the average gain or absorption of a finite bandwidth beam."""
        start_frequency = center_frequency - frequency_bandwidth / 2
        end_frequency = center_frequency + frequency_bandwidth / 2
        start_value = spectrum_func(start_frequency, mode_field_radius)
        middle_value = spectrum_func(center_frequency, mode_field_radius)
        end_value = spectrum_func(end_frequency, mode_field_radius)
        return np.mean([start_value, middle_value, end_value])

    def plot_gain_and_absorption_spectrum(self):
        """Convenience plotting function to draw the imported cross section data and the calculated interpolates to
        check that they match."""
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
        plt.show()
