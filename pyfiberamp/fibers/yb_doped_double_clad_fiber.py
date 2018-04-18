from .double_clad_fiber import DoubleCladFiber
from pyfiberamp.spectroscopies import YbGermanoSilicate

class YbDopedDoubleCladFiber(DoubleCladFiber):
    """YbDopedDoubleCladFiber is a convenience class for Yb-doped double-clad fiber that uses the default spectroscopic
     data for Yb-ions."""
    def __init__(self, length, core_radius, ion_number_density,
                 background_loss, core_na, ratio_of_core_and_cladding_diameters):
        """

        :param length: Fiber length
        :type length: float
        :param core_radius: Core radius
        :type core_radius: float
        :param ion_number_density: Number density of the dopant ions (1/m^3)
        :type ion_number_density: float
        :param background_loss: Linear loss of the core (1/m, NOT in dB/m)
        :type background_loss: float
        :param core_na: Numerical aperture of the core
        :type core_na: float
        :param ratio_of_core_and_cladding_diameters: Core diameter divided by cladding diameter
        :type ratio_of_core_and_cladding_diameters: float

        """
        super().__init__(length=length,
                         core_radius=core_radius,
                         ion_number_density=ion_number_density,
                         spectroscopy=YbGermanoSilicate,
                         background_loss=background_loss,
                         core_na=core_na,
                         ratio_of_core_and_cladding_diameters=ratio_of_core_and_cladding_diameters)