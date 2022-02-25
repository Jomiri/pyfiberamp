from abc import ABC, abstractmethod

from pyfiberamp.doping_profile import DopingProfile
from pyfiberamp.helper_funcs import *
from pyfiberamp.mode_solver.lp_mode_solver import default_mode_solver


class FiberBase(ABC):
    """ FiberBase is the base class for the other fiber classes. It contains methods for calculating
    nonlinear effective area, mode field diameter and setting up optical channels. In principle, PassiveFiber class
    could act as a base class but subclassing ActiveFiber from PassiveFiber feels conceptually wrong."""

    @abstractmethod
    def __init__(self, length=0, core_radius=0, background_loss=0, core_na=0):
        """

        :param length: Fiber length
        :type length: float
        :param core_radius: Core radius
        :type core_radius: float
        :param background_loss: Linear loss of the core (1/m, NOT in dB/m)
        :type background_loss: float
        :param core_na: Numerical aperture of the core
        :type core_na: float

        """
        self.length = length
        self.core_radius = core_radius
        self.background_loss = background_loss
        self.core_na = core_na
        self.core_refractive_index = DEFAULT_GROUP_INDEX
        self.effective_area_type = 'mode'
        self.doping_profile = DopingProfile(ion_number_densities=[0], radii=[core_radius],
                                            num_of_angular_sections=1, core_radius=core_radius)

    def default_signal_mode(self, freq):
        return default_mode_solver.find_mode(l=0, m=1,
                                             core_radius=self.core_radius, na=self.core_na, wl=freq_to_wl(freq))

    def default_pump_mode(self, freq):
        return self.default_signal_mode(freq)

    def v_parameter(self, wl):
        return fiber_v_parameter(wl, self.core_radius, self.core_na)

    @property
    def num_ion_populations(self):
        return len(self.doping_profile.ion_number_densities)

    def core_area(self):
        """Returns the core area of the fiber defined as pi*r**2, where r is the core radius.

        :returns: Core area
        :rtype: float

        """
        return self.core_radius**2 * np.pi

    @abstractmethod
    def get_channel_emission_cross_section(self, freq, frequency_bandwidth):
        pass

    @abstractmethod
    def get_channel_absorption_cross_section(self, freq, frequency_bandwidth):
        pass

    @abstractmethod
    def saturation_parameter(self):
        pass

