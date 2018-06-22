from .helper_funcs import *
from pyfiberamp.util.mode_shape import ModeShape


class OpticalChannel:
    def __init__(self, v, dv, input_power, direction, overlaps, gain, absorption, loss, label,
                 reflection_target_label, reflection_coeff, channel_type):
        self.v = v
        self.dv = dv
        self.input_power = input_power
        self.direction = direction
        self.overlaps = overlaps
        self.gain = gain
        self.absorption = absorption
        self.loss = loss
        self.peak_power_func = lambda x: x
        self.number_of_modes = NUMBER_OF_MODES_IN_SINGLE_MODE_FIBER
        self.label = label
        self.reflection_target_label = reflection_target_label
        self.reflection_coeff = reflection_coeff
        self.channel_type = channel_type

    @property
    def wavelength(self):
        return freq_to_wl(self.v)

    @classmethod
    def create_signal_channel(cls, fiber, wl, wl_bandwidth, power, mode_shape_parameters, direction, label,
                              reflection_target_label, reflectance, channel_type=''):

        mode_shape_parameters = cls.fill_mode_shape_parameters(mode_shape_parameters,
                                                               fiber.default_signal_mode_shape_parameters)

        return cls._create_channel(fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                                   direction, label, reflection_target_label, reflectance,
                                   channel_type)

    @classmethod
    def create_pump_channel(cls, fiber, wl, wl_bandwidth, power, mode_shape_parameters, direction, label,
                            reflection_target_label, reflectance, channel_type=''):

        mode_shape_parameters = cls.fill_mode_shape_parameters(mode_shape_parameters,
                                                               fiber.default_pump_mode_shape_parameters)

        return cls._create_channel(fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                                   direction, label, reflection_target_label, reflectance,
                                   channel_type)

    @staticmethod
    def fill_mode_shape_parameters(input_parameters, default_parameters):
        if input_parameters is None:
            return default_parameters
        return {**default_parameters, **input_parameters}

    @classmethod
    def _create_channel(cls, fiber, wl, wl_bandwidth, power, mode_shape_parameters,
                        direction, label, reflection_target_label,
                        reflection_coeff, channel_type):

        n_ion_populations = fiber.num_ion_populations
        overlaps = cls.get_overlaps(fiber, wl, mode_shape_parameters)
        center_frequency = wl_to_freq(wl)
        frequency_bandwidth = wl_bw_to_freq_bw(wl_bandwidth, wl)
        gain = overlaps * fiber.get_channel_emission_cross_section(center_frequency, frequency_bandwidth) * fiber.doping_profile.ion_number_densities
        absorption = overlaps * fiber.get_channel_absorption_cross_section(center_frequency, frequency_bandwidth) * fiber.doping_profile.ion_number_densities
        center_frequency = np.full(n_ion_populations, center_frequency)
        frequency_bandwidth = np.full(n_ion_populations, frequency_bandwidth)
        loss = np.full(n_ion_populations, fiber.background_loss)
        return OpticalChannel(center_frequency, frequency_bandwidth, power,
                              direction, overlaps, gain, absorption, loss,
                              label, reflection_target_label, reflection_coeff,
                              channel_type)

    @staticmethod
    def get_overlaps(fiber, wl, mode_shape_parameters):
        # Case 1: overlaps predefined
        n_preset_overlaps = len(mode_shape_parameters['overlaps'])
        if n_preset_overlaps > 0:
            assert n_preset_overlaps == len(fiber.doping_profile.areas)
            return np.array(mode_shape_parameters['overlaps'])

        # No overlaps defined -> fiber must specify doping profile radii for overlap calculation
        doping_radii = fiber.doping_profile.radii
        assert len(doping_radii) > 0

        # Case 2: Mode shape and overlaps must be calculated
        mode_shape = ModeShape(fiber, wl, mode_shape_parameters)
        overlaps = mode_shape.get_ring_overlaps(doping_radii)
        return overlaps


