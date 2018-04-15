import numpy as np
from itertools import chain
from functools import partial

from .helper_funcs import *
from .sliced_array import SlicedArray


class DelayedExecutor:
    def __init__(self):
        self.funcs_and_args = []

    def add_func(self, func, args):
        self.funcs_and_args.append((func, args))

    def execute(self):
        for func, args in self.funcs_and_args:
            func(*args)

    def reset(self):
        self.funcs_and_args = []


class Channels:
    def __init__(self):
        self.fiber = None
        self.delayed_executor = DelayedExecutor()

        self.forward_signals = []
        self.backward_signals = []

        self.forward_pumps = []
        self.backward_pumps = []

        self.forward_ase = []
        self.backward_ase = []

        self.forward_ramans = []
        self.backward_ramans = []

    def set_fiber(self, fiber):
        self.fiber = fiber
        self.refresh()

    def refresh(self):
        self.forward_signals = []
        self.backward_signals = []
        self.forward_pumps = []
        self.backward_pumps = []
        self.forward_ase = []
        self.backward_ase = []
        self.forward_ramans = []
        self.backward_ramans = []
        self.delayed_executor.execute()

    def add_forward_signal(self, wl, power, mfd):
        self.delayed_executor.add_func(self._init_forward_signal, (wl, power, mfd))

    def _init_forward_signal(self, wl, power, mfd):
        channel = self.fiber.create_in_core_forward_single_frequency_channel(wl, power, mfd)
        self.forward_signals.append(channel)

    def add_pulsed_forward_signal(self, wl, power, f_rep, fwhm_duration, mfd):
        self.delayed_executor.add_func(self._init_pulsed_forward_signal, (wl, power, f_rep, fwhm_duration, mfd))

    def _init_pulsed_forward_signal(self, wl, power, f_rep, fwhm_duration, mfd):
        channel = self.fiber.create_in_core_forward_single_frequency_channel(wl, power, mfd)
        channel.peak_power_func = partial(gaussian_peak_power, (f_rep, fwhm_duration))
        check_signal_reprate(f_rep)
        self.forward_signals.append(channel)

    def add_backward_signal(self, wl, power, mfd):
        self.delayed_executor.add_func(self._init_backward_signal, (wl, power, mfd))

    def _init_backward_signal(self, wl, power, mfd):
        channel = self.fiber.create_in_core_backward_single_frequency_channel(wl, power, mfd)
        self.backward_signals.append(channel)

    def add_forward_pump(self, wl, power, mfd):
        self.delayed_executor.add_func(self._init_forward_pump, (wl, power, mfd))

    def _init_forward_pump(self, wl, power, mfd):
        channel = self.fiber.create_forward_pump_channel(wl, power, mfd)
        self.forward_pumps.append(channel)

    def add_backward_pump(self, wl, power, mfd):
        self.delayed_executor.add_func(self._init_backward_pump, (wl, power, mfd))

    def _init_backward_pump(self, wl, power, mfd):
        channel = self.fiber.create_backward_pump_channel(wl, power, mfd)
        self.backward_pumps.append(channel)

    def add_ase(self, wl_start, wl_end, n_bins):
        self.delayed_executor.add_func(self._init_ase, (wl_start, wl_end, n_bins))

    def _init_ase(self, wl_start, wl_end, n_bins):
        assert (wl_end > wl_start)
        assert (isinstance(n_bins, int) and n_bins > 0)
        ase_wl_bandwidth = (wl_end - wl_start) / n_bins
        ase_wls = np.linspace(wl_start, wl_end, n_bins)
        for wl in ase_wls:
            forward_channel = self.fiber.create_in_core_forward_finite_bandwidth_channel(wl, ase_wl_bandwidth,
                                                                                         SIMULATION_MIN_POWER, 0)
            self.forward_ase.append(forward_channel)

            backward_channel = self.fiber.create_in_core_backward_finite_bandwidth_channel(wl, ase_wl_bandwidth,
                                                                                           SIMULATION_MIN_POWER, 0)
            self.backward_ase.append(backward_channel)

    def add_raman(self, input_power, backward_raman_allowed):
        self.delayed_executor.add_func(self._init_raman, (input_power, backward_raman_allowed))

    def _init_raman(self, input_power, backward_raman_allowed):
        assert len(self.forward_signals) == 1 and len(self.backward_signals) == 0, 'Raman modeling is supported only ' \
                                                                                   'with a single forward signal.'
        raman_freq = self.forward_signals[0].v - RAMAN_FREQ_SHIFT
        raman_wl = freq_to_wl(raman_freq)
        forward_channel = self.fiber.create_in_core_forward_finite_bandwidth_channel(raman_wl, RAMAN_GAIN_WL_BANDWIDTH,
                                                                                     input_power, 0)
        backward_channel = self.fiber.create_in_core_backward_finite_bandwidth_channel(raman_wl,
                                                                                       RAMAN_GAIN_WL_BANDWIDTH,
                                                                                       input_power, 0)
        forward_channel.number_of_modes = RAMAN_MODES_IN_PM_FIBER
        backward_channel.number_of_modes = RAMAN_MODES_IN_PM_FIBER
        if not backward_raman_allowed:
            backward_channel.dv = 0
        self.forward_ramans.append(forward_channel)
        self.backward_ramans.append(backward_channel)

    def get_wavelengths(self):
        return freq_to_wl(self.get_frequencies())

    def get_frequencies(self):
        return self._to_sliced_array([ch.v for ch in self._all_channels()])

    def get_frequency_bandwidths(self):
        return self._to_sliced_array([ch.dv for ch in self._all_channels()])

    def get_propagation_directions(self):
        return self._to_sliced_array([ch.direction for ch in self._all_channels()])

    def get_number_of_modes(self):
        return self._to_sliced_array([ch.number_of_modes for ch in self._all_channels()])

    def get_absorption(self):
        return self._to_sliced_array([ch.absorption for ch in self._all_channels()])

    def get_gain(self):
        return self._to_sliced_array([ch.gain for ch in self._all_channels()])

    def get_background_loss(self):
        return self._to_sliced_array([ch.loss for ch in self._all_channels()])

    def get_input_powers(self):
        return self._to_sliced_array([ch.input_power for ch in self._all_channels()])

    def _all_channels(self):
        return chain(self.forward_signals, self.backward_signals, self.forward_pumps, self.backward_pumps,
                     self.forward_ase, self.backward_ase, self.forward_ramans, self.backward_ramans)

    def _to_sliced_array(self, iterable):
        return SlicedArray(np.array(iterable), self.get_slices())

    def get_slices(self):
        n_forward_signal = len(self.forward_signals)
        n_backward_signal = len(self.backward_signals)
        n_forward_pump = len(self.forward_pumps)
        n_backward_pump = len(self.backward_pumps)
        n_ase = len(self.forward_ase)
        n_raman = len(self.forward_ramans)

        backward_signal_start = n_forward_signal
        forward_pump_start = backward_signal_start + n_backward_signal
        backward_pump_start = forward_pump_start + n_forward_pump
        forward_ase_start = backward_pump_start + n_backward_pump
        backward_ase_start = forward_ase_start + n_ase
        forward_raman_start = backward_ase_start + n_ase
        backward_raman_start = forward_raman_start + n_raman
        backward_raman_end = backward_raman_start + n_raman

        slices = {'forward_signal': slice(0, backward_signal_start),
                  'backward_signal': slice(backward_signal_start, forward_pump_start),
                  'forward_pump': slice(forward_pump_start, backward_pump_start),
                  'backward_pump': slice(backward_pump_start, forward_ase_start),
                  'forward_ase': slice(forward_ase_start, backward_ase_start),
                  'backward_ase': slice(backward_ase_start, forward_raman_start),
                  'forward_raman': slice(forward_raman_start, backward_raman_start),
                  'backward_raman': slice(backward_raman_start, backward_raman_end)}
        return slices

    @property
    def backward_raman_allowed(self):
        return len(self.backward_ramans) != 0 and self.backward_ramans[0].dv != 0
