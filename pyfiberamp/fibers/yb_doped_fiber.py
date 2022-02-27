from . import ActiveFiber
from pyfiberamp.spectroscopies import YbGermanoSilicate


class YbDopedFiber(ActiveFiber):
    """YbDopedFiber is a convenience class for Yb-doped single-mode fiber that uses the default spectroscopic data
     for Yb-ions."""

    def __init__(self, length=0, core_radius=0, ion_number_density=0, background_loss=0, core_na=0):
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

        """
        super().__init__(length=length,
                         core_radius=core_radius,
                         background_loss=background_loss,
                         core_na=core_na,
                         ion_number_density=ion_number_density,
                         spectroscopy=YbGermanoSilicate)
