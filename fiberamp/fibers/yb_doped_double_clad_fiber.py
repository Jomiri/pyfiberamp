from .double_clad_fiber import DoubleCladFiber
from fiberamp.helper_funcs import YB_ABSORPTION_CS_FILE, YB_EMISSION_CS_FILE, YB_UPPER_STATE_LIFETIME


class YbDopedDoubleCladFiber(DoubleCladFiber):
    def __init__(self, length, core_r, ion_number_density,
                 background_loss, core_NA, core_to_cladding_ratio):
        super().__init__(length,
                         YB_ABSORPTION_CS_FILE,
                         YB_EMISSION_CS_FILE,
                         core_r,
                         YB_UPPER_STATE_LIFETIME,
                         ion_number_density,
                         background_loss,
                         core_NA,
                         core_to_cladding_ratio)