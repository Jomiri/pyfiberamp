from .active_fiber import *


class DoubleCladFiber(ActiveFiber):
    def __init__(self, length, absorption_cs_file, gain_cs_file,
                 core_r, upper_state_lifetime, ion_number_density,
                 background_loss, core_NA, core_to_cladding_ratio):
        super().__init__(length, absorption_cs_file, gain_cs_file,
                         core_r, upper_state_lifetime,
                         ion_number_density, background_loss,
                         core_NA)
        self.core_to_cladding_ratio = core_to_cladding_ratio

    def pump_to_core_overlap(self):
        return self.core_to_cladding_ratio**2

    def pump_cladding_radius(self):
        return self.core_r / self.core_to_cladding_ratio

    def make_overlap(self, freq, mode_field_radii, slices):
        overlap = super().make_overlap(freq, mode_field_radii, slices)
        if self.overlap_is_constant:
            overlap = np.full_like(freq, overlap)
        overlap[slices['co_pump_slice']] = self.pump_to_core_overlap()
        overlap[slices['counter_pump_slice']] = self.pump_to_core_overlap()
        overlap.shape = freq.shape
        return overlap

    def get_pump_channel_gain(self, freq):
        return self.gain_cs_interp(freq) * self.pump_to_core_overlap() * self.ion_number_density

    def get_pump_channel_absorption(self, freq):
        return self.absorption_cs_interp(freq) * self.pump_to_core_overlap() * self.ion_number_density

    def create_forward_pump_channel(self, wl, power, preset_mfd):
        return self.create_cladding_pump_channel(wl, power, direction=+1)

    def create_backward_pump_channel(self, wl, power, preset_mfd):
        return self.create_cladding_pump_channel(wl, power, direction=-1)

    def create_cladding_pump_channel(self, wl, power, direction):
        frequency = wl_to_freq(wl)
        frequency_bandwidth = 0
        mfd = self.pump_cladding_radius()
        gain = self.get_pump_channel_gain(frequency)
        absorption = self.get_pump_channel_absorption(frequency)
        loss = self.background_loss
        return OpticalChannel(frequency, frequency_bandwidth, power, direction, mfd, gain, absorption, loss)


