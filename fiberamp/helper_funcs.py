import numpy as np
import warnings

from .parameters import *


def load_two_column_file(file_name):
    return np.loadtxt(file_name, converters={0: to_float, 1: to_float})


def wl_bw_to_freq_bw(wl_bw, center_wl):
    return c/center_wl**2 * wl_bw


def wl_to_freq(wl):
    return c/wl


def freq_to_wl(f):
    return c/f


def to_float(x):
    x = x.decode()
    x = x.replace(',', '.')
    return float(x)


def decibel_to_exp(x):
    return x / (np.log10(np.e) * 10)


def exp_to_decibel(x):
    return x * np.log10(np.e) * 10


def to_db(x):
    return 10 * np.log10(x)


def fiber_v_parameter(wl, r, na):
    return 2 * np.pi / wl * r * na


def fundamental_mode_mfd_marcuse(wl, r, na):
    # Marcuse's equation
    v = fiber_v_parameter(wl, r, na)
    return 2 * r * (0.65 + 1.619*v**(-3/2) + 2.879*v**(-6))


def fundamental_mode_mfd_petermann_2(wl, r, na):
    # Petermann II equation
    v = fiber_v_parameter(wl, r, na)
    return 2 * r * (0.65 + 1.619*v**(-3/2) + 2.879*v**(-6) - (0.015 + 1.561*v**(-7)))


def fundamental_mode_radius(wl, r, na):
    return fundamental_mode_mfd_petermann_2(wl, r, na) / 2


def overlap_integral(doped_radius, mode_radius):
    return 1 - np.exp(-doped_radius**2 / mode_radius**2)


def overlap_from_wl(wl, r, na, doped_radius):
    mode_radius = fundamental_mode_radius(wl, r, na)
    return overlap_integral(doped_radius, mode_radius)


def overlap_from_freq(freq, r, na, doped_radius):
    return overlap_from_wl(freq_to_wl(freq), r, na, doped_radius)


def effective_area_from_mfd(wl, r, na):
    half_width_at_e = fundamental_mode_radius(wl, r, na)
    area = np.pi * half_width_at_e**2
    return area


def eta_from_fiber_parameters(core_r, tau, ion_number_density):
    return np.pi * core_r**2 * ion_number_density / tau


def gaussian_peak_power(average_power, f_rep, fwhm_duration):
    pulse_energy = average_power / f_rep
    peak_power = 2 * np.sqrt(np.log(2)) / np.sqrt(np.pi) * pulse_energy / fwhm_duration
    return peak_power


def linspace_2d(start_arr, end_arr, length):
    diff = end_arr - start_arr
    return start_arr[:, np.newaxis] + np.arange(length) * diff[:, np.newaxis] / (length - 1)


def check_signal_reprate(f_rep, f_rep_lower_limit):
    if f_rep < f_rep_lower_limit:
        warnings.warn('Signal with repetition rate of {:.1f} Hz cannot be treated as quasi-continuous.'.format(f_rep))
