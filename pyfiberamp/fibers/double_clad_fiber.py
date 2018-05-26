from .active_fiber import *
from pyfiberamp.optical_channel import OpticalChannel


class DoubleCladFiber(ActiveFiber):
    """DoubleCladFiber extends ActiveFiber and describes a double-clad active fiber with single-mode step-index core.
    Flat-top pump distribution in the pump cladding is assumed as well as constant overlap between the pump modes and
    the core. All pump beams propagate in the pump cladding."""
    @classmethod
    def from_cross_section_files(cls, length=0, absorption_cs_file=None, emission_cs_file=None,
                 core_radius=0, upper_state_lifetime=0, ion_number_density=0,
                 background_loss=0, core_na=0, ratio_of_core_and_cladding_diameters=0):
        """
        :param length: Fiber length
        :type length: float
        :param absorption_cs_file: Name of the file containing absorption cross-section data
        :type absorption_cs_file: str
        :param emission_cs_file: Name of the file containing emission cross-section data
        :type emission_cs_file: str
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

        spectroscopy = Spectroscopy.from_files(absorption_cs_file, emission_cs_file, upper_state_lifetime)
        cls(length, core_radius, background_loss, core_na, spectroscopy, ion_number_density,
            ratio_of_core_and_cladding_diameters)

    def __init__(self, length=0, core_radius=0, background_loss=0, core_na=0,
                 spectroscopy=None, ion_number_density=0, ratio_of_core_and_cladding_diameters=0):

        """
        :param length: Fiber length
        :type length: float
        :param core_radius: Core radius
        :type core_radius: float
        :param background_loss: Linear loss of the core (1/m, NOT in dB/m)
        :type background_loss: float
        :param core_na: Numerical aperture of the core
        :type core_na: float
        :param spectroscopy: The spectroscopic properties of the fiber.
        :type spectroscopy: :class:`~pyfiberamp.spectroscopies.Spectroscopy`
        :param ion_number_density: Number density of the dopant ions (1/m^3)
        :type ion_number_density: float
        :param ratio_of_core_and_cladding_diameters: Core diameter divided by cladding diameter
        :type ratio_of_core_and_cladding_diameters: float

        """
        super().__init__(length=length,
                         core_radius=core_radius,
                         spectroscopy=spectroscopy,
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

    def get_pump_channel_gain(self, freq, frequency_bandwidth, mode_field_radius):
        if frequency_bandwidth == 0:
            return self._single_frequency_pump_channel_gain(freq)
        else:
            return self._finite_bandwidth_pump_channel_gain(freq, frequency_bandwidth)

    def get_pump_channel_absorption(self, freq, frequency_bandwidth, mode_field_radius):
        if frequency_bandwidth == 0:
            return self._single_frequency_pump_channel_absorption(freq)
        else:
            return self._finite_bandwidth_pump_channel_absorption(freq, frequency_bandwidth)

    def _single_frequency_pump_channel_gain(self, freq):
        """This is the maximum gain g* defined in the Giles model. The gain for a mode with a given frequency depends
        on the the emission cross section, overlap between mode and core/ions (here ratio of core and cladding areas)
        and the doping concentration."""
        return self.spectroscopy.gain_cs_interp(freq) * self.pump_to_core_overlap() * self.ion_number_density

    def _single_frequency_pump_channel_absorption(self, freq):
        """This is the maximum absorption alpha defined in the Giles model. The absorption for a mode with a given
         frequency depends on the the absorption cross section, overlap between mode and core/ions (here ratio of core
         and cladding areas) and the doping concentration."""
        return self.spectroscopy.absorption_cs_interp(freq) * self.pump_to_core_overlap() * self.ion_number_density

    def _finite_bandwidth_pump_channel_gain(self, center_frequency, frequency_bandwidth):
        """Calculates the maximum gain g* for a finite bandwidth signal by averaging over the start, end and middle
        points."""
        return self._averaged_value_of_finite_bandwidth_pump_spectrum(center_frequency, frequency_bandwidth,
                                                                      self._single_frequency_pump_channel_gain)

    def _finite_bandwidth_pump_channel_absorption(self, center_frequency, frequency_bandwidth):
        """Calculates the maximum absorption alpha for a finite bandwidth signal by averaging over the start, end
        and middle points."""
        return self._averaged_value_of_finite_bandwidth_pump_spectrum(center_frequency, frequency_bandwidth,
                                                                      self._single_frequency_pump_channel_absorption)

    def _averaged_value_of_finite_bandwidth_pump_spectrum(self, center_frequency, frequency_bandwidth, spectrum_func):
        """Actual function used to calculate the average gain or absorption of a finite bandwidth beam."""
        start_frequency = center_frequency - frequency_bandwidth / 2
        end_frequency = center_frequency + frequency_bandwidth / 2
        start_value = spectrum_func(start_frequency)
        middle_value = spectrum_func(center_frequency)
        end_value = spectrum_func(end_frequency)
        return np.mean([start_value, middle_value, end_value])