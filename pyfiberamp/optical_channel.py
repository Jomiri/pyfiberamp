from .helper_funcs import *


class OpticalChannel:
    def __init__(self, v, dv, input_power, direction, mfd, gain, absorption, loss, label,
                 reflection_target_label, reflection_coeff, channel_type):
        self.v = v
        self.dv = dv
        self.input_power = input_power
        self.direction = direction
        self.mfd = mfd
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
    def create_signal_channel(cls, fiber, wl, wl_bandwidth, power, preset_mfd, direction, label,
                              reflection_target_label, reflection_coeff, channel_type=''):
        return cls._create_channel(fiber, wl , wl_bandwidth, power, preset_mfd, direction, label, reflection_target_label,
                               reflection_coeff, fiber.get_signal_channel_gain, fiber.get_signal_channel_absorption,
                                   channel_type)



    @classmethod
    def create_pump_channel(cls, fiber, wl, wl_bandwidth, power, preset_mfd, direction, label,
                              reflection_target_label, reflection_coeff, channel_type=''):
        return cls._create_channel(fiber, wl , wl_bandwidth, power, preset_mfd, direction, label, reflection_target_label,
                                   reflection_coeff, fiber.get_pump_channel_gain, fiber.get_pump_channel_absorption,
                                   channel_type)

    @classmethod
    def _create_channel(cls, fiber, wl, wl_bandwidth, power, preset_mfd, direction, label, reflection_target_label,
                        reflection_coeff, gain_func, absorption_func, channel_type):
        center_frequency = wl_to_freq(wl)
        mfd = fiber._mode_field_diameter_for_channel(center_frequency, preset_mfd)
        mode_field_radius = mfd / 2
        frequency_bandwidth = wl_bw_to_freq_bw(wl_bandwidth, wl)
        gain = gain_func(center_frequency, frequency_bandwidth, mode_field_radius)
        absorption = absorption_func(center_frequency, frequency_bandwidth, mode_field_radius)
        loss = fiber.background_loss
        return OpticalChannel(center_frequency, frequency_bandwidth, power,
                              direction, mfd, gain, absorption, loss, label, reflection_target_label, reflection_coeff,
                              channel_type)

