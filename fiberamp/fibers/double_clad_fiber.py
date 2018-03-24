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


