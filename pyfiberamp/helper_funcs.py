"""
.. module:: helper_funcs
    :synopsis: Contains short utility functions needed by other modules.

.. moduleauthor:: Joona Rissanen

"""
import numpy as np
import warnings
from scipy.interpolate import interp1d

from pyfiberamp.parameters import *


def load_spectrum(file_name):
    """Loads a spectrum file with two columns of floats as numpy array. The first column is wavelength in nanometers;
    the second column is some spectroscopic property (mostly cross section) in SI units."""
    spectrum = load_two_column_file(file_name)
    spectrum[:, 0] *= 1e-9
    return spectrum


def load_two_column_file(file_name):
    """Loads a file with two columns of floats as a numpy array."""
    return np.loadtxt(file_name, converters={0: to_float, 1: to_float})


def to_float(x):
    x = x.decode()
    x = x.replace(',', '.')
    return float(x)


def wl_bw_to_freq_bw(wl_bw, center_wl):
    """Transforms a spectral bandwidth in wavelength centered at wavelength center_wl
    into a spectral bandwidth in frequency.

    :param wl_bw: Wavelength bandwidth
    :type name: float or numpy array of floats
    :param center_wl: Central wavelength of the spectrum
    :type center_wl: float or numpy array of floats
    :returns: Frequency bandwidth
    :rtype: float or numpy array
    """
    return c/center_wl**2 * wl_bw


def wl_to_freq(wl):
    """Transforms (vacuum) wavelength to frequency."""
    return c/wl


def freq_to_wl(f):
    """Transforms frequency to (vacuum) wavelength."""
    return c/f


def decibel_to_exp(x):
    """Transforms a logarithmic quantity from dB/m to 1/m."""
    return x / (np.log10(np.e) * 10)


def exp_to_decibel(x):
    """Transforms a logarithmic quantity from 1/m to dB/m."""
    return x * np.log10(np.e) * 10


def to_db(x):
    """Transforms a quantity to decibels."""
    return 10 * np.log10(x)


def to_dbm(power):
    """Transforms a power in Watts to dBm."""
    return to_db(power / 1000)


def overlap_from_freq(freq, r, na, doped_radius):
    """Calculates the overlap factor of a fundamental fiber mode with frequency freq and the doped core.

    :param freq: Frequency of the mode
    :type freq: float
    :param r: Core radius
    :type r: float
    :param na: Core numerical aperture
    :type na: float
    :param doped_radius: Core radius
    :type doped_radius: float
    :returns: Overlap factor
    :rtype: float
    """
    return overlap_from_wl(freq_to_wl(freq), r, na, doped_radius)


def overlap_from_wl(wl, r, na, doped_radius):
    """Calculates the overlap factor of a fundamental fiber mode with wavelength wl and the doped core.

    :param wl: Wavelength of the mode
    :type wl: float
    :param r: Core radius
    :type r: float
    :param na: Core numerical aperture
    :type na: float
    :param doped_radius: Core radius
    :type doped_radius: float
    :returns: Overlap factor
    :rtype: float
    """
    mode_radius = fundamental_mode_radius_petermann_2(wl, r, na)
    return overlap_integral(doped_radius, mode_radius)


def effective_area_from_mfd(wl, r, na):
    """Calculates an approximation of the nonlinear effective area of the fiber as pi*(mfd/2)**2.

    :param wl: Wavelength of the mode
    :type wl: float
    :param r: Core radius
    :type r: float
    :param na: Core numerical aperture
    :type na: float
    :returns: Nonlinear effective area
    :rtype: float
    """
    half_width_at_e = fundamental_mode_radius_petermann_2(wl, r, na)
    return np.pi * half_width_at_e**2


def fundamental_mode_mfd_marcuse(wl, r, na):
    """Calculates the mode field diameter of the fundamental mode with vacuum wavelength wl using Marcuse's equation.

    :param wl: Wavelength of the mode
    :type wl: float
    :param r: Core radius
    :type r: float
    :param na: Core numerical aperture
    :type na: float
    :returns: Mode field diameter of the fundamental mode
    :rtype: float
    """
    v = fiber_v_parameter(wl, r, na)
    return 2 * r * (0.65 + 1.619*v**(-3/2) + 2.879*v**(-6))


def fundamental_mode_mfd_petermann_2(wl, r, na):
    """Calculates the mode field diameter of the fundamental mode with vacuum wavelength wl using the Petermann II
    equation.

    :param wl: Wavelength of the mode
    :type wl: float
    :param r: Core radius
    :type r: float
    :param na: Core numerical aperture
    :type na: float
    :returns: Mode field diameter of the fundamental mode
    :rtype: float
    """
    v = fiber_v_parameter(wl, r, na)
    return 2 * r * (0.65 + 1.619*v**(-3/2) + 2.879*v**(-6) - (0.015 + 1.561*v**(-7)))


def fundamental_mode_radius_petermann_2(wl, r, na):
    """Calculates the fundamental mode radius with vacuum wavelength wl using the Petermann II equation.

    :param wl: Wavelength of the mode
    :type wl: float
    :param r: Core radius
    :type r: float
    :param na: Core numerical aperture
    :type na: float
    :returns: Mode field radius of the fundamental mode
    :rtype: float
    """
    return fundamental_mode_mfd_petermann_2(wl, r, na) / 2


def overlap_integral(doped_radius, mode_radius):
    """Overlap integral between the Gaussian-shaped fundamental mode (approximation) and rectangular doped core.

    :param doped_radius: Radius of the dopant in the fiber (typically core radius)
    :type doped_radius: float
    :param mode_radius: Mode field radius of the propagating optical beam
    :type mode_radius: float
    :returns: Overlap integral between the mode and the dopant ions
    :rtype: float
    """
    return 1 - np.exp(-doped_radius**2 / mode_radius**2)


def fiber_v_parameter(wl, r, na):
    """Calculates the V-parameter or normalized frequency of a fiber mode with vacuum wavelength wl.

    :param wl: Wavelength of the mode
    :type wl: float
    :param r: Core radius
    :type r: float
    :param na: Core numerical aperture
    :type na: float
    :returns: V-parameter of the mode
    :rtype: float
    """
    return 2 * np.pi / wl * r * na


def zeta_from_fiber_parameters(core_radius, upper_state_lifetime, ion_number_density):
    """Calculates the Giles modes saturation parameter zeta.

    :param core_radius: Core radius of the fiber
    :type core_radius: float
    :param upper_state_lifetime: Lifetime of the excited state
    :type upper_state_lifetime: float
    :param ion_number_density: Number density of the dopant ions (1/m^3)
    :type ion_number_density: float
    :returns: Saturation parameter zeta
    :rtype: float
    """
    return np.pi * core_radius**2 * ion_number_density / upper_state_lifetime


def gaussian_peak_power(average_power, f_rep, fwhm_duration):
    """Calculates the peak power of a Gaussian pulse.

    :param average_power: Average power of the pulse signal
    :type average_power: float
    :param f_rep: Repetition rate of the pulsed signal
    :type f_rep: float
    :param fwhm_duration: FWHM duration of the Gaussian pulses
    :type fwhm_duration: float
    :returns: Peak power of the pulses
    :rtype: float
    """
    pulse_energy = average_power / f_rep
    peak_power = 2 * np.sqrt(np.log(2)) / np.sqrt(np.pi) * pulse_energy / fwhm_duration
    return peak_power


def resample_array(arr, N):
    """Changes the width of an array to N columns by using linear interpolation to each row.
    :param arr: Array to be resized
    :type arr: 2D numpy array
    :param N: Number of columns in the resized array
    :type N: int
    :returns: The resized array with N colums.
    :rtype: 2D numpy array
    """
    x_original = np.arange(arr.shape[1])
    x_new = np.linspace(0, x_original[-1], N)
    interpolant = interp1d(x_original, arr)
    arr_new = interpolant(x_new)
    return arr_new


def linspace_2d(start_vec, end_vec, length):
    """Creates a numpy array with given start and end vectors as first and last columns and a total number of columns
    specified by "length". The middle columns are linearly interpolated.

    :param start_vec: First column of the generated array
    :type start_vec: 1D numpy array
    :param end_vec: Last column of the generated array
    :type end_vec: 1D numpy array
    :param length: Total number of columns in the generated array
    :returns: Array interpolated between the start and end vectors
    :rtype: 2D numpy array
    """
    diff = end_vec - start_vec
    return start_vec[:, np.newaxis] + np.arange(length) * diff[:, np.newaxis] / (length - 1)


def check_signal_reprate(f_rep):
    """Emits a warning if the repetition rate of the signal is too low to be accurately modelled due to pulse-to-pulse
    gain variations.

    :param f_rep: Repetition frequency
    :type f_rep: float
    """
    if f_rep < REP_RATE_LOWER_LIMIT:
        warnings.warn('Signal with repetition rate of {:.1f} Hz cannot be treated as quasi-continuous.'.format(f_rep))
