from .fiber_base import FiberBase


class PassiveFiber(FiberBase):
    def __init__(self, length, core_r, background_loss, core_na):
        super().__init__(length, core_r, background_loss, core_na)

    def saturation_parameter(self):
        return 1

    def get_channel_gain(self, freq, mode_field_radius):
        return 0

    def get_channel_absorption(self, freq, mode_field_radius):
        return 0

    def finite_bandwidth_gain(self, center_frequency, frequency_bandwidth, mode_field_radius):
        return 0

    def finite_bandwidth_absorption(self, center_frequency, frequency_bandwidth, mode_field_radius):
        return 0

