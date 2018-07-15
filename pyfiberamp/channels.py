from functools import partial
from itertools import chain

from pyfiberamp.util.sliced_array import SlicedArray
from .helper_funcs import *
from pyfiberamp.util import DelayedExecutor
from pyfiberamp.optical_channel import OpticalChannel


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

    def add_forward_signal(self, *args):
        self.delayed_executor.add_func(self._init_forward_signal, args)

    def add_pulsed_forward_signal(self, *args):
        self.delayed_executor.add_func(self._init_pulsed_forward_signal, args)

    def add_backward_signal(self, *args):
        self.delayed_executor.add_func(self._init_backward_signal, args)

    def add_forward_pump(self, *args):
        self.delayed_executor.add_func(self._init_forward_pump, args)

    def add_backward_pump(self, *args):
        self.delayed_executor.add_func(self._init_backward_pump, args)

    def add_ase(self, wl_start, wl_end, n_bins):
        self.delayed_executor.add_func(self._init_ase, (wl_start, wl_end, n_bins))

    def add_raman(self, input_power, backward_raman_allowed):
        self.delayed_executor.add_func(self._init_raman, (input_power, backward_raman_allowed))

    def _init_forward_signal(self, wl, wl_bandwidth, power, mode_shape_parameters, label,
                             reflection_target_label="", reflection_coeff=0):
        channel = OpticalChannel.create_signal_channel(self.fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                                                       direction=1, label=label,
                                                       reflection_target_label=reflection_target_label,
                                                       reflectance=reflection_coeff,
                                                       channel_type='forward_signal')
        self.forward_signals.append(channel)

    def _init_pulsed_forward_signal(self, wl, wl_bandwidth, power, f_rep, fwhm_duration, mode_shape_parameters, label,
                             reflection_target_label="", reflection_coeff=0):
        channel = OpticalChannel.create_signal_channel(self.fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                                                       direction=1, label=label,
                                                       reflection_target_label=reflection_target_label,
                                                       reflectance=reflection_coeff,
                                                       channel_type='forward_signal')
        channel.peak_power_func = partial(gaussian_peak_power, (f_rep, fwhm_duration))
        check_signal_reprate(f_rep)
        self.forward_signals.append(channel)

    def _init_backward_signal(self, wl, wl_bandwidth, power, mode_shape_parameters, label,
                             reflection_target_label="", reflection_coeff=0):
        channel = OpticalChannel.create_signal_channel(self.fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                                                       direction=-1, label=label,
                                                       reflection_target_label=reflection_target_label,
                                                       reflectance=reflection_coeff,
                                                       channel_type='backward_signal')
        self.backward_signals.append(channel)

    def _init_backward_pump(self, wl, wl_bandwidth, power, mode_shape_parameters, label,
                                  reflection_target_label="", reflection_coeff=0):
        channel = OpticalChannel.create_pump_channel(self.fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                                                     direction=-1, label=label,
                                                     reflection_target_label=reflection_target_label,
                                                     reflectance=reflection_coeff,
                                                     channel_type='backward_pump')
        self.backward_pumps.append(channel)

    def _init_forward_pump(self, wl, wl_bandwidth, power, mode_shape_parameters, label,
                            reflection_target_label="", reflection_coeff=0):
        channel = OpticalChannel.create_pump_channel(self.fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                                                     direction=1, label=label,
                                                     reflection_target_label=reflection_target_label,
                                                     reflectance=reflection_coeff,
                                                     channel_type='forward_pump')
        self.forward_pumps.append(channel)

    def _init_ase(self, wl_start, wl_end, n_bins):
        assert wl_end > wl_start, 'End wavelength must be greater than start wavelength.'
        assert isinstance(n_bins, int) and n_bins > 0, 'Number of ASE bins must be a positive integer.'
        ase_wl_bandwidth = (wl_end - wl_start) / n_bins
        ase_wls = np.linspace(wl_start, wl_end, n_bins)
        for wl in ase_wls:
            forward_channel = OpticalChannel.create_signal_channel(self.fiber, wl, ase_wl_bandwidth,
                                                                   SIMULATION_MIN_POWER, None, direction=1, label="",
                                                                   reflection_target_label='', reflectance=0,
                                                                   channel_type='forward_ase')
            self.forward_ase.append(forward_channel)

            backward_channel = OpticalChannel.create_signal_channel(self.fiber, wl, ase_wl_bandwidth,
                                                                    SIMULATION_MIN_POWER, None, direction=-1, label="",
                                                                    reflection_target_label='', reflectance=0,
                                                                    channel_type='backward_ase')
            self.backward_ase.append(backward_channel)

    def _init_raman(self, input_power, backward_raman_allowed):
        assert len(self.forward_signals) == 1 and len(self.backward_signals) == 0, 'Raman modeling is supported only ' \
                                                                                   'with a single forward signal.'
        raman_freq = self.forward_signals[0].v - RAMAN_FREQ_SHIFT
        raman_wl = freq_to_wl(raman_freq)
        forward_channel = OpticalChannel.create_signal_channel(self.fiber, raman_wl, RAMAN_GAIN_WL_BANDWIDTH,
                                                               input_power, None, direction=1, label="",
                                                               reflection_target_label='', reflectance=0,
                                                               channel_type='forward_raman')
        backward_channel = OpticalChannel.create_signal_channel(self.fiber, raman_wl, RAMAN_GAIN_WL_BANDWIDTH,
                                                                input_power, None, direction=-1, label="",
                                                                reflection_target_label='', reflectance=0,
                                                                channel_type='backward_raman')
        forward_channel.number_of_modes = RAMAN_MODES_IN_PM_FIBER
        backward_channel.number_of_modes = RAMAN_MODES_IN_PM_FIBER
        if not backward_raman_allowed:
            backward_channel.dv = 0
        self.forward_ramans.append(forward_channel)
        self.backward_ramans.append(backward_channel)

    def get_wavelengths(self):
        return freq_to_wl(self.get_frequencies())

    def get_frequencies(self):
        return self._to_sliced_array(np.hstack([ch.v for ch in self._all_channels()]))

    def get_frequency_bandwidths(self):
        return self._to_sliced_array(np.hstack([ch.dv for ch in self._all_channels()]))

    def get_propagation_directions(self):
        return self._to_sliced_array(np.hstack([ch.direction for ch in self._all_channels()]))

    def get_number_of_modes(self):
        return self._to_sliced_array(np.hstack([ch.number_of_modes for ch in self._all_channels()]))

    def get_absorption(self):
        return self._to_sliced_array(np.hstack([ch.absorption for ch in self._all_channels()]))

    def get_gain(self):
        return self._to_sliced_array(np.hstack([ch.gain for ch in self._all_channels()]))

    def get_background_loss(self):
        return self._to_sliced_array(np.hstack([ch.loss for ch in self._all_channels()]))

    def get_input_powers(self):
        return self._to_sliced_array(np.hstack([ch.input_power for ch in self._all_channels()]))

    def get_labels(self):
        all_labels = np.array(np.hstack([ch.label for ch in self._all_channels()]))
        valid_labels = all_labels[all_labels!='']
        _, counts = np.unique(valid_labels, return_counts=True)
        assert np.all(counts == 1), 'Multiple channels have the same label!'
        return all_labels

    def get_reflections(self):
        reflection_list = [(ch.label, ch.reflection_target_label, ch.reflection_coeff) for ch in self._all_channels()
                           if ch.reflection_target_label != '']

        return self._translate_reflection_labels(reflection_list)

    def _translate_reflection_labels(self, reflections):
        processed = []
        for r in reflections:
            source_index = self.get_label_index(r[0])
            target_index = self.get_label_index(r[1])
            R = r[2]
            processed.append((source_index, target_index, R))
        return processed

    def get_dynamic_input_powers(self, max_time_steps):
        input_array = np.zeros((self.number_of_channels, max_time_steps+1))
        for ch_idx, ch in enumerate(self._all_channels()):
            input_power = ch.input_power
            if isinstance(input_power, float):
                input = np.ones(max_time_steps + 1) * input_power
            else:
                input = np.zeros(max_time_steps + 1)
                input[0:-1] = input_power
                input[-1] = input_power[-1]
            input_array[ch_idx, :] = input

        min_clamp(input_array, SIMULATION_MIN_POWER)
        return input_array

    def _all_channels(self):
        return chain(self.forward_signals, self.forward_pumps, self.forward_ase, self.forward_ramans,
                     self.backward_signals, self.backward_pumps, self.backward_ase, self.backward_ramans)

    def _to_sliced_array(self, iterable):
        return SlicedArray(np.array(iterable), self.get_slices())

    def get_slices(self):
        n_forward_signal = len(self.forward_signals)
        n_backward_signal = len(self.backward_signals)
        n_forward_pump = len(self.forward_pumps)
        n_backward_pump = len(self.backward_pumps)
        n_ase = len(self.forward_ase)
        n_raman = len(self.forward_ramans)

        forward_pump_start = n_forward_signal
        forward_ase_start = forward_pump_start + n_forward_pump
        forward_raman_start = forward_ase_start + n_ase
        backward_signal_start = forward_raman_start + n_raman
        backward_pump_start = backward_signal_start + n_backward_signal
        backward_ase_start = backward_pump_start + n_backward_pump
        backward_raman_start = backward_ase_start + n_ase
        backward_raman_end = backward_raman_start + n_raman

        slices = {'forward_signal': slice(0, forward_pump_start),
                  'forward_pump': slice(forward_pump_start, forward_ase_start),
                  'forward_ase': slice(forward_ase_start, forward_raman_start),
                  'forward_raman': slice(forward_raman_start, backward_signal_start),
                  'backward_signal': slice(backward_signal_start, backward_pump_start),
                  'backward_pump': slice(backward_pump_start, backward_ase_start),
                  'backward_ase': slice(backward_ase_start, backward_raman_start),
                  'backward_raman': slice(backward_raman_start, backward_raman_end)}
        return slices

    def get_forward_and_backward_slices(self):
        n_forward = len(self.forward_signals) + len(self.forward_pumps)\
                    + len(self.forward_ase) + len(self.forward_ramans)
        n_backward = len(self.backward_signals) + len(self.backward_pumps)\
                     + len(self.backward_ase) + len(self.backward_ramans)
        return slice(0, n_forward), slice(n_forward, n_forward + n_backward)

    @property
    def backward_raman_allowed(self):
        return len(self.backward_ramans) != 0 and self.backward_ramans[0].dv != 0

    @property
    def number_of_channels(self):
        return len(list(self._all_channels()))

    def get_label_index(self, label):
        if isinstance(label, int):
            assert (0 <= label < self.number_of_channels)
            return label
        elif isinstance(label, str):
            all_labels = self.get_labels()
            return int(np.where(all_labels == label)[0])

