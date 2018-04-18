from pyfiberamp.helper_funcs import *

from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


class Spectroscopy:
    @classmethod
    def from_files(cls, absorption_cross_section_file, emission_cross_section_file, upper_state_lifetime):
        absorption_spectrum = load_spectrum(absorption_cross_section_file)
        gain_spectrum = load_spectrum(emission_cross_section_file)
        return cls(absorption_spectrum, gain_spectrum, upper_state_lifetime)

    def __init__(self, absorption_cross_sections, emission_cross_sections, upper_state_lifetime):
        self.absorption_cs_spectrum = absorption_cross_sections
        self.emission_cs_spectrum = emission_cross_sections
        self.absorption_cs_interp = self._make_cross_section_interpolate(absorption_cross_sections)
        self.gain_cs_interp = self._make_cross_section_interpolate(emission_cross_sections)
        self.upper_state_lifetime = upper_state_lifetime

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

    def plot_gain_and_absorption_spectrum(self):
        """Convenience plotting function to draw the imported cross section data and the calculated interpolates to
        check that they match."""
        fig, ax = plt.subplots()
        gain = self.emission_cs_spectrum
        absorption = self.absorption_cs_spectrum
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


YbGermanoSilicate= Spectroscopy.from_files(YB_ABSORPTION_CS_FILE, YB_EMISSION_CS_FILE, YB_UPPER_STATE_LIFETIME)
