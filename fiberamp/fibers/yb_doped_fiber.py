from . import ActiveFiber
from ..helper_funcs import YB_ABSORPTION_CS_FILE, YB_EMISSION_CS_FILE, YB_UPPER_STATE_LIFETIME


class YbDopedFiber(ActiveFiber):
    def __init__(self, length, core_r, ion_number_density, background_loss, core_na):
        super().__init__(length,
                         YB_ABSORPTION_CS_FILE,
                         YB_EMISSION_CS_FILE,
                         core_r,
                         YB_UPPER_STATE_LIFETIME,
                         ion_number_density,
                         background_loss,
                         core_na)