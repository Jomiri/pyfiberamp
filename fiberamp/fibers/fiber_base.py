from abc import ABC, abstractmethod
import numpy as np

from fiberamp.helper_funcs import *
from fiberamp.optical_channel import OpticalChannel


class FiberBase:
    @abstractmethod
    def __init__(self, length, core_r, background_loss, core_na):
        self.length = length
        self.core_r = core_r
        self.background_loss = background_loss
        self.core_na = core_na
        self.effective_area_type = 'core_area'

    def nonlinear_effective_area(self, freq):
        if self.effective_area_type == 'core_area':
            return self.effective_area_from_core_area(freq)
        elif self.effective_area_type == 'gaussian':
            return self.effective_area_from_gaussian_approximation(freq)
        elif self.effective_area_type == 'bessel':
            return self.effective_area_from_bessel_distribution(freq)

    def effective_area_from_core_area(self, freq):
        return self.core_area() * np.ones_like(freq)

    def core_area(self):
        return self.core_r ** 2 * np.pi

    def effective_area_from_gaussian_approximation(self, freq):
        return effective_area_from_mfd(freq_to_wl(freq), self.core_r, self.core_na)

    def effective_area_from_bessel_distribution(self, freq):
        raise NotImplementedError

    def get_mfd(self, freq, preset_mfd):
        if preset_mfd > 0:
            return preset_mfd
        else:
            return fundamental_mode_mfd_petermann_2(freq_to_wl(freq), self.core_r, self.core_na)

    def create_in_core_single_frequency_channel(self, wl, power, preset_mfd, direction):
        frequency = wl_to_freq(wl)
        frequency_bandwidth = 0
        mfd = self.get_mfd(frequency, preset_mfd)
        mode_field_radius = mfd / 2
        gain = self.get_channel_gain(frequency, mode_field_radius)
        absorption = self.get_channel_absorption(frequency, mode_field_radius)
        loss = self.background_loss
        return OpticalChannel(frequency, frequency_bandwidth, power, direction, mfd, gain, absorption, loss)

    def create_in_core_finite_bandwidth_channel(self, wl, wl_bandwidth, power, preset_mfd, direction):
        center_frequency = wl_to_freq(wl)
        frequency_bandwidth = wl_bw_to_freq_bw(wl_bandwidth, wl)
        mfd = self.get_mfd(center_frequency, preset_mfd)
        mode_field_radius = mfd / 2
        gain = self.finite_bandwidth_gain(center_frequency, frequency_bandwidth, mode_field_radius)
        absorption = self.finite_bandwidth_absorption(center_frequency, frequency_bandwidth, mode_field_radius)
        loss = self.background_loss
        return OpticalChannel(center_frequency, frequency_bandwidth, power, direction, mfd, gain, absorption, loss)

    def create_in_core_forward_single_frequency_channel(self, wl, power, preset_mfd):
        return self.create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=+1)

    def create_in_core_backward_single_frequency_channel(self, wl, power, preset_mfd):
        return self.create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=-1)

    def create_in_core_forward_finite_bandwidth_channel(self, wl, wl_bandwidth, power, preset_mfd):
        return self.create_in_core_finite_bandwidth_channel(wl, wl_bandwidth, power, preset_mfd, direction=+1)

    def create_in_core_backward_finite_bandwidth_channel(self, wl, wl_bandwidth, power, preset_mfd):
        return self.create_in_core_finite_bandwidth_channel(wl, wl_bandwidth, power, preset_mfd, direction=-1)


    @abstractmethod
    def get_channel_gain(self, freq, mode_field_radius):
        pass

    @abstractmethod
    def get_channel_absorption(self, freq, mode_field_radius):
        pass

    @abstractmethod
    def finite_bandwidth_gain(self, center_frequency, frequency_bandwidth, mode_field_radius):
        pass

    @abstractmethod
    def finite_bandwidth_absorption(self, center_frequency, frequency_bandwidth, mode_field_radius):
        pass

    @abstractmethod
    def saturation_parameter(self):
        pass
