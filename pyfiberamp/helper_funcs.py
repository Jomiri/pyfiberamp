import numpy as np
import warnings

from .parameters import *


def load_spectrum(file_name):
    """Loads a spectrum file with two columns of floats as numpy array. The first column is wavelength in nanometers;
    the second column is in SI units."""
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
    """Transforms a spectral bandwidth in nanometers centered at wavelength center_wl
    into a spectral bandwidth in Hz."""
    return c/center_wl**2 * wl_bw


def wl_to_freq(wl):
    return c/wl


def freq_to_wl(f):
    return c/f


def decibel_to_exp(x):
    return x / (np.log10(np.e) * 10)


def exp_to_decibel(x):
    return x * np.log10(np.e) * 10


def to_db(x):
    return 10 * np.log10(x)


def overlap_from_freq(freq, r, na, doped_radius):
    """Calculates the overlap factor of a fundamental fiber mode with frequency freq and the doped core.
    Parameters
    -----------
    freq : float
        Frequency of the mode
    r : float
        Core radius
    na : float
        Core numerical aperture
    doped_radius : float
        Core radius
    """
    return overlap_from_wl(freq_to_wl(freq), r, na, doped_radius)


def overlap_from_wl(wl, r, na, doped_radius):
    """Calculates the overlap factor of a fundamental fiber mode with wavelength wl and the doped core.
    Parameters
    -----------
    wl : float
        Wavelength
    r : float
        Core radius
    na : float
        Core numerical aperture
    doped_radius : float
        Core radius
    """
    mode_radius = fundamental_mode_radius_petermann_2(wl, r, na)
    return overlap_integral(doped_radius, mode_radius)


def effective_area_from_mfd(wl, r, na):
    """Calculates an approximation of the nonlinear effective area of the fiber as pi*(mfd/2)**2.
    Parameters
    -----------
    wl : float
        Wavelength
    r : float
        Core radius
    na : float
        Core numerical aperture
    """
    half_width_at_e = fundamental_mode_radius_petermann_2(wl, r, na)
    return np.pi * half_width_at_e**2


def fundamental_mode_mfd_marcuse(wl, r, na):
    """Calculates the mode field diameter of the fundamental mode with vacuum wavelength wl using Marcuse's equation.
    Parameters
    -----------
    wl : float
        Wavelength
    r : float
        Core radius
    na : float
        Core numerical aperture
    """
    v = fiber_v_parameter(wl, r, na)
    return 2 * r * (0.65 + 1.619*v**(-3/2) + 2.879*v**(-6))


def fundamental_mode_mfd_petermann_2(wl, r, na):
    """Calculates the mode field diameter of the fundamental mode with vacuum wavelength wl using the Petermann II
    equation.
    Parameters
    -----------
    wl : float
        Wavelength
    r : float
        Core radius
    na : float
        Core numerical aperture
    """
    v = fiber_v_parameter(wl, r, na)
    return 2 * r * (0.65 + 1.619*v**(-3/2) + 2.879*v**(-6) - (0.015 + 1.561*v**(-7)))


def fundamental_mode_radius_petermann_2(wl, r, na):
    """Calculates the fundamental mode radius with vacumm wavelength wl using the Petermann II equation.
    Parameters
    -----------
    wl : float
        Wavelength
    r : float
        Core radius
    na : float
        Core numerical aperture
    """
    return fundamental_mode_mfd_petermann_2(wl, r, na) / 2


def overlap_integral(doped_radius, mode_radius):
    """Overlap integral between the Gaussian-shaped fundamental mode (approximation) and rectangular doped core."""
    return 1 - np.exp(-doped_radius**2 / mode_radius**2)


def fiber_v_parameter(wl, r, na):
    """Calculates the V-parameter or normalized frequency of a fiber mode with vacuum wavelength wl.
    Parameters
    -----------
    wl : float
        Wavelength
    r : float
        Core radius
    na : float
        Core numerical aperture
    """
    return 2 * np.pi / wl * r * na


def zeta_from_fiber_parameters(core_radius, upper_state_lifetime, ion_number_density):
    """Calculates the Giles modes saturation parameter zeta.    """
    return np.pi * core_radius**2 * ion_number_density / upper_state_lifetime


def gaussian_peak_power(average_power, f_rep, fwhm_duration):
    """Calculates the peak power of a Gaussian pulse.
    Parameters
    -----------
    average_power : float
        Time average power of the pulse train
    f_rep : float
        Pulse repetition frequency
    fwhm_duration : float
        Full-width at half-maximum pulse duration.
    """
    pulse_energy = average_power / f_rep
    peak_power = 2 * np.sqrt(np.log(2)) / np.sqrt(np.pi) * pulse_energy / fwhm_duration
    return peak_power


def linspace_2d(start_arr, end_arr, length):
    """Creates a numpy array with given start and end vectors as first and last columns and a total number of columns
    specified by "length". The middle columns are linearly interpolated.
    Parameters
    -----------
    start_arr : numpy array
        First column
    end_arr : numpy array
        Last column
    length : int
        Total number of columns
    """
    diff = end_arr - start_arr
    return start_arr[:, np.newaxis] + np.arange(length) * diff[:, np.newaxis] / (length - 1)


def check_signal_reprate(f_rep):
    """Emits a warning if the repetition rate of the signal is too low to be accurately modelled due to pulse-to-pulse
    gain variations.
    Parameters
    ----------
    f_rep : float
        Repetition frequency
    """
    if f_rep < REP_RATE_LOWER_LIMIT:
        warnings.warn('Signal with repetition rate of {:.1f} Hz cannot be treated as quasi-continuous.'.format(f_rep))
