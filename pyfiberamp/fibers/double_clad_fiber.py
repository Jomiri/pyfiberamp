from .active_fiber import *
from pyfiberamp.optical_channel import OpticalChannel


class DoubleCladFiber(ActiveFiber):
    """DoubleCladFiber extends ActiveFiber and describes a double-clad active fiber with single-mode step-index core.
    Flat-top pump distribution in the pump cladding is assumed as well as constant overlap between the pump modes and
    the core. All pump beams propagate in the pump cladding."""
    def __init__(self, length=0, absorption_cs_file=None, gain_cs_file=None,
                 core_radius=0, upper_state_lifetime=0, ion_number_density=0,
                 background_loss=0, core_na=0, ratio_of_core_and_cladding_diameters=0):
        """

        :param length: Fiber length
        :type length: float
        :param absorption_cs_file: Name of the file containing absorption cross-section data
        :type absorption_cs_file: str
        :param gain_cs_file: Name of the file containing emission cross-section data
        :type gain_cs_file: str
        :param core_radius: Core radius
        :type core_radius: float
        :param upper_state_lifetime: Lifetime of the excited state
        :type upper_state_lifetime: float
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
                         absorption_cs_file=absorption_cs_file,
                         gain_cs_file=gain_cs_file,
                         core_radius=core_radius,
                         upper_state_lifetime=upper_state_lifetime,
                         ion_number_density=ion_number_density,
                         background_loss=background_loss,
                         core_na=core_na)
        self.core_to_cladding_ratio = ratio_of_core_and_cladding_diameters

    def pump_to_core_overlap(self):
        """Returns the overlap between the core and the pump beams, which equals to the ratio of core and cladding
        area."""
        return self.core_to_cladding_ratio**2

    def pump_cladding_radius(self):
        """Returns the radius of the fiber's pump cladding."""
        return self.core_radius / self.core_to_cladding_ratio

    def create_forward_pump_channel(self, wl, power, preset_mfd):
        """Wrapper method for "create_cladding_pump_channel" to hide the direction parameter."""
        return self._create_cladding_pump_channel(wl, power, direction=+1)

    def create_backward_pump_channel(self, wl, power, preset_mfd):
        """Wrapper method for "create_cladding_pump_channel" to hide the direction parameter."""
        return self._create_cladding_pump_channel(wl, power, direction=-1)

    def _create_cladding_pump_channel(self, wl, power, direction):
        """Creates an OpticalChannel object describing a cladding pump beam."""
        frequency = wl_to_freq(wl)
        frequency_bandwidth = 0
        mfd = self.pump_cladding_radius()
        gain = self._get_pump_channel_gain(frequency)
        absorption = self._get_pump_channel_absorption(frequency)
        loss = self.background_loss
        return OpticalChannel(frequency, frequency_bandwidth, power, direction, mfd, gain, absorption, loss)

    def _get_pump_channel_gain(self, freq):
        """This is the maximum gain g* defined in the Giles model. The gain for a mode with a given frequency depends
        on the the emission cross section, overlap between mode and core/ions (here ratio of core and cladding areas)
        and the doping concentration."""
        return self.gain_cs_interp(freq) * self.pump_to_core_overlap() * self.ion_number_density

    def _get_pump_channel_absorption(self, freq):
        """This is the maximum absorption alpha defined in the Giles model. The absorption for a mode with a given
         frequency depends on the the absorption cross section, overlap between mode and core/ions (here ratio of core
         and cladding areas) and the doping concentration."""
        return self.absorption_cs_interp(freq) * self.pump_to_core_overlap() * self.ion_number_density
