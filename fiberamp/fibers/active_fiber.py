from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from fiberamp.helper_funcs import *

CROSS_SECTION_SMOOTHING_FACTOR = 1e-51
SPECTRUM_PLOT_NPOINTS = 1000


class ActiveFiber:
    def __init__(self, length, absorption_cs_file, gain_cs_file,
                 core_r, upper_state_lifetime, ion_number_density,
                 background_loss, core_na):

        absorption_spectrum = load_two_column_file(absorption_cs_file)
        gain_spectrum = load_two_column_file(gain_cs_file)

        absorption_spectrum[:, 0] *= 1e-9
        gain_spectrum[:, 0] *= 1e-9
        self.absorption_spectrum = absorption_spectrum
        self.gain_spectrum = gain_spectrum
        self.absorption_cs_interp = None
        self.gain_cs_interp = None
        self.make_interpolates()

        self.length = length
        self.core_r = core_r
        self.tau = upper_state_lifetime
        self.ion_number_density = ion_number_density
        self.background_loss = background_loss
        self.core_na = core_na
        self.overlap = None
        self.overlap_function = None
        self.use_automatic_overlap()
        self.effective_area_type = 'core_area'

    @property
    def overlap_is_constant(self):
        return self.overlap is not None

    def make_interpolates(self):
        self.absorption_cs_interp = self.make_interpolate(self.absorption_spectrum)
        self.gain_cs_interp = self.make_interpolate(self.gain_spectrum)

    @staticmethod
    def make_interpolate(spectrum):
        frequency = wl_to_freq(spectrum[::-1, 0])
        cross_section = spectrum[::-1, 1]
        spline = UnivariateSpline(frequency, cross_section, s=CROSS_SECTION_SMOOTHING_FACTOR, ext='zeros')

        def interp(freq):
            cross_sec = spline(freq)
            cross_sec [cross_sec < 0] = 0
            return cross_sec

        return interp

    def use_constant_overlap(self, overlap):
        self.overlap = overlap
        self.overlap_function = None

    def use_automatic_overlap(self):
        self.overlap_function = lambda f: overlap_from_freq(f, r=self.core_r, na=self.core_na, doped_radius=self.core_r)
        self.overlap = None

    def make_gain_spectrum(self, freq, mode_field_radii, slices):
        return self.gain_cs_interp(freq) * self.make_overlap(freq, mode_field_radii, slices) * self.ion_number_density

    def make_absorption_spectrum(self, freq, mode_field_radii, slices):
        return (self.absorption_cs_interp(freq) * self.make_overlap(freq, mode_field_radii, slices)
                * self.ion_number_density)

    def make_overlap(self, freq, mode_field_radii, slices):
        if self.overlap_is_constant:
            return self.overlap
        overlap = self.overlap_function(freq)
        predefined_radii_mask = mode_field_radii > 0
        overlap[predefined_radii_mask] = overlap_integral(self.core_r, mode_field_radii[predefined_radii_mask])
        return overlap

    def eta(self):
        return eta_from_fiber_parameters(self.core_r, self.tau, self.ion_number_density)

    def nonlinear_effective_area(self, freq):
        if self.effective_area_type == 'core_area':
            return self.effective_area_from_core_area(freq)
        elif self.effective_area_type == 'gaussian':
            return self.effective_area_from_gaussian_approximation(freq)
        elif self.effective_area_type == 'bessel':
            return self.effective_area_from_bessel_distribution(freq)

    def effective_area_from_core_area(self, freq):
        return self.core_area() * np.ones_like(freq)

    def core_area(self):
        return self.core_r**2 * np.pi

    def effective_area_from_gaussian_approximation(self, freq):
        return effective_area_from_mfd(freq_to_wl(freq), self.core_r, self.core_na)

    def effective_area_from_bessel_distribution(self, freq):
        raise NotImplementedError

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
