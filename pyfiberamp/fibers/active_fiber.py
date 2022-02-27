from pyfiberamp.helper_funcs import *
from .fiber_base import FiberBase
from pyfiberamp.spectroscopies import Spectroscopy
from pyfiberamp.fibers.doping_profile import DopingProfile


class ActiveFiber(FiberBase):
    """ActiveFiber describes a step-index single-mode fiber with active dopant ions. Currently, only uniform doping
    in the whole core area is supported. This class extends the FiberBase class by adding spectroscopic data: gain and
    emission spectra, upper state lifetime and doping concentration."""
    @classmethod
    def from_cross_section_files(cls, length, absorption_cs_file=None, emission_cs_file=None,
                     core_radius=0, upper_state_lifetime=0, ion_number_density=0,
                     background_loss=0, core_na=0):
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
        """
        spectroscopy = Spectroscopy.from_files(absorption_cs_file, emission_cs_file, upper_state_lifetime)
        return cls(length, core_radius, background_loss, core_na, spectroscopy, ion_number_density)

    def __init__(self, length=0, core_radius=0, background_loss=0, core_na=0,
                 spectroscopy=None, ion_number_density=0):
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

        """
        super().__init__(length=length,
                         core_radius=core_radius,
                         background_loss=background_loss,
                         core_na=core_na)

        self.spectroscopy = spectroscopy
        self.doping_profile = DopingProfile(ion_number_densities=[ion_number_density], radii=[core_radius],
                                            num_of_angular_sections=1, core_radius=core_radius)

    def set_doping_profile(self, ion_number_densities, radii=None, num_angular_sections=1):
        if radii is None:
            radii = [self.core_radius]
        self.doping_profile = DopingProfile(ion_number_densities, radii, num_angular_sections, self.core_radius)

    def set_ion_number_density_based_on_core_absorption(self, wl: float, absorption: float):
        cross_section = self.get_channel_absorption_cross_section(wl_to_freq(wl), 0.0)
        core_mode = self.default_signal_mode(wl_to_freq(wl))
        overlap = core_mode.core_overlap
        ion_number_density = decibel_to_exp(absorption) / (overlap * cross_section)
        self.doping_profile.ion_number_densities = np.full_like(self.doping_profile.ion_number_densities,
                                                                ion_number_density)

    @property
    def ion_number_density(self):
        assert len(self.doping_profile.ion_number_densities) == 1
        return self.doping_profile.ion_number_densities[0]

    def saturation_parameter(self):
        """Returns the constant saturation parameter zeta defined in the Giles model."""
        return zeta_from_fiber_parameters(self.core_radius, self.spectroscopy.upper_state_lifetime, self.ion_number_density)

    def get_channel_emission_cross_section(self, freq, frequency_bandwidth):
        if frequency_bandwidth == 0:
            return self.spectroscopy.gain_cs_interp(freq)
        else:
            return averaged_value_of_finite_bandwidth_spectrum(freq, frequency_bandwidth, self.spectroscopy.gain_cs_interp)

    def get_channel_absorption_cross_section(self, freq, frequency_bandwidth):
        if frequency_bandwidth == 0:
            return self.spectroscopy.absorption_cs_interp(freq)
        else:
            return averaged_value_of_finite_bandwidth_spectrum(freq, frequency_bandwidth, self.spectroscopy.absorption_cs_interp)
