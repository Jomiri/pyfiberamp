from .double_clad_fiber import DoubleCladFiber
from pyfiberamp.helper_funcs import YB_ABSORPTION_CS_FILE, YB_EMISSION_CS_FILE, YB_UPPER_STATE_LIFETIME


class YbDopedDoubleCladFiber(DoubleCladFiber):
    """YbDopedDoubleCladFiber is a convenience class for Yb-doped double-clad fiber that uses the default spectroscopic
     data for Yb-ions."""
    def __init__(self, length, core_radius, ion_number_density,
                 background_loss, core_na, ratio_of_core_and_cladding_diameters):
        """Parameters
        -------------

        length : float
            Fiber length
        core_radius : float
            Core radius
        ion_number_density : float
            Yb doping concentration (1/m^3)
        background_loss : float
            Linear loss of the fiber (1/m, not dB/m)
        core_na : float
            Numerical aperture of the core
        ratio_of_core_and_cladding_diameters : float
            -
        """
        super().__init__(length=length,
                         absorption_cs_file=YB_ABSORPTION_CS_FILE,
                         gain_cs_file=YB_EMISSION_CS_FILE,
                         core_radius=core_radius,
                         upper_state_lifetime=YB_UPPER_STATE_LIFETIME,
                         ion_number_density=ion_number_density,
                         background_loss=background_loss,
                         core_na=core_na,
                         ratio_of_core_and_cladding_diameters=ratio_of_core_and_cladding_diameters)