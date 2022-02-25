import matplotlib.pyplot as plt

from pyfiberamp.util.sliced_array import SlicedArray
from pyfiberamp.helper_funcs import *
import pyfiberamp.plotting.plotter as plotter


class SimulationResult:
    def __init__(self, z, powers, upper_level_fraction, channels, fiber, backward_raman_allowed=True):
        self.z = z
        slices = channels.get_slices()
        self.channels = channels
        self.channel_ids = channels.get_channel_ids()
        self.powers = SlicedArray(powers, slices)
        self.wavelengths = channels.get_wavelengths()
        self.upper_level_fraction = upper_level_fraction
        self._backward_raman_allowed = backward_raman_allowed
        self.fiber = fiber
        self.use_db_scale = False

    @property
    def local_average_excitation(self):
        weights = self.fiber.doping_profile.areas
        norm_weights = weights[:, np.newaxis] / sum(weights) * len(weights)
        return np.mean(self.upper_level_fraction * norm_weights, axis=0)

    @property
    def overall_average_excitation(self):
        return np.mean(self.local_average_excitation)

    def make_result_dict(self):
        result_dict = {}
        for idx, ch in enumerate(self.channels.as_iterator()):
            ch_power = self.powers[idx, :]
            start_idx, end_idx = (0, -1) if ch.direction == 1 else (-1, 0)
            output_power = ch_power[end_idx]
            input_power = ch_power[start_idx]
            gain = to_db(output_power / input_power)
            result_dict[ch.channel_id] = {'input_power': input_power,
                                          'output_power': output_power,
                                          'gain': gain}
        return result_dict

    def powers_at_fiber_end(self):
        forward_slice, backward_slice = self.channels.get_forward_and_backward_slices()
        return np.hstack((self.powers[forward_slice, -1], self.powers[backward_slice,0]))

    def plot(self, figsize=DEFAULT_FIGSIZE):
        plotter.plot_simulation_result(self, figsize)

    def plot_power_evolution(self, figsize=DEFAULT_FIGSIZE):
        plotter.plot_power_evolution(self, figsize)





