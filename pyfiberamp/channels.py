from itertools import chain, count

from pyfiberamp.util.sliced_array import SlicedArray
from .helper_funcs import *
from pyfiberamp.optical_channel import OpticalChannel


class Channels:
    def __init__(self, fiber):
        self.fiber = fiber
        self.id_generator = count(0, 1)

        self.forward_signals = []
        self.backward_signals = []

        self.forward_pumps = []
        self.backward_pumps = []

        self.forward_ase = []
        self.backward_ase = []

        self.forward_ramans = []
        self.backward_ramans = []

    def check_channel_id(self, channel_id):
        existing_ids = [] if self.number_of_channels == 0 else self.get_channel_ids()
        if channel_id is None:
            channel_id = next(self.id_generator)
            while channel_id in list(existing_ids):
                channel_id = next(self.id_generator)
        elif channel_id in existing_ids:
            raise RuntimeError(f'Channel id {channel_id} already in use!')
        return channel_id

    def create_channel(self, channel_type, direction,
                       fiber, input_power, wl, mode, channel_id=None,
                       num_of_modes=NUMBER_OF_ASE_POLARIZATION_MODES, wl_bandwidth=0.0,
                       loss=None,
                       reflection_target_id=None, reflectance=0.0,
                       peak_power_func=lambda x: x):
        if mode is None:
            if channel_type in ['signal', 'ase', 'raman']:
                mode = fiber.default_signal_mode(wl_to_freq(wl))
            else:
                mode = fiber.default_pump_mode(wl_to_freq(wl))

        channel = OpticalChannel.from_mode(channel_id, channel_type, direction,
                                           fiber, input_power, wl, mode,
                                           num_of_modes, wl_bandwidth,
                                           loss,
                                           reflection_target_id, reflectance,
                                           peak_power_func)
        self.add_channel(channel)

    def add_channel(self, channel):
        channel.channel_id = self.check_channel_id(channel.channel_id)

        if channel.channel_type == 'signal':
            if channel.direction == 1:
                self.forward_signals.append(channel)
            elif channel.direction == -1:
                self.backward_signals.append(channel)
        elif channel.channel_type == 'pump':
            if channel.direction == 1:
                self.forward_pumps.append(channel)
            elif channel.direction == -1:
                self.backward_pumps.append(channel)
        elif channel.channel_type == 'ase':
            if channel.direction == 1:
                self.forward_ase.append(channel)
            elif channel.direction == -1:
                self.backward_ase.append(channel)
        elif channel.channel_type == 'raman':
            if channel.direction == 1:
                self.forward_ramans.append(channel)
            elif channel.direction == -1:
                self.backward_ramans.append(channel)
        else:
            raise RuntimeError(f'Channel type {channel.channel_type} not supported')

    def get_channel_ids(self):
        return self._to_sliced_array((np.hstack([ch.channel_id for ch in self._all_channels()])))

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

    def get_reflections(self):
        reflection_list = [(ch.channel_id, ch.reflection_target_id, ch.end_reflection_coeff)
                           for ch in self._all_channels() if ch.reflection_target_id is not None]

        return self._translate_reflection_ids(reflection_list)

    def _translate_reflection_ids(self, reflections):
        processed = []
        for r in reflections:
            source_index = self.get_channel_id_index(r[0])
            target_index = self.get_channel_id_index(r[1])
            R = r[2]
            processed.append((source_index, target_index, R))
        return processed

    def get_dynamic_input_powers(self, max_time_steps):
        input_array = np.zeros((self.number_of_channels, max_time_steps+1))
        for ch_idx, ch in enumerate(self._all_channels()):
            input_power = ch.input_power
            if isinstance(input_power, float) or isinstance(input_power, int):
                channel_input = np.ones(max_time_steps + 1) * input_power
            else:
                channel_input = np.zeros(max_time_steps + 1)
                channel_input[0:-1] = input_power
                channel_input[-1] = input_power[-1]
            input_array[ch_idx, :] = channel_input

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
        n_forward_ase = len(self.forward_ase)
        n_backward_ase = len(self.backward_ase)
        n_raman = len(self.forward_ramans)

        forward_pump_start = n_forward_signal
        forward_ase_start = forward_pump_start + n_forward_pump
        forward_raman_start = forward_ase_start + n_forward_ase
        backward_signal_start = forward_raman_start + n_raman
        backward_pump_start = backward_signal_start + n_backward_signal
        backward_ase_start = backward_pump_start + n_backward_pump
        backward_raman_start = backward_ase_start + n_backward_ase
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
        n_forward = len(self.forward_signals) + len(self.forward_pumps) \
                    + len(self.forward_ase) + len(self.forward_ramans)
        n_backward = len(self.backward_signals) + len(self.backward_pumps) \
                     + len(self.backward_ase) + len(self.backward_ramans)
        return slice(0, n_forward), slice(n_forward, n_forward + n_backward)

    @property
    def backward_raman_allowed(self):
        return len(self.backward_ramans) != 0 and self.backward_ramans[0].dv != 0

    @property
    def number_of_channels(self):
        return len(list(self._all_channels()))

    def get_channel_id_index(self, channel_id):
        return int(np.where(self.get_channel_ids() == channel_id)[0])

