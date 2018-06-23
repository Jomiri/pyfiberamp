from abc import ABC, abstractmethod

from pyfiberamp.doping_profile import DopingProfile
from pyfiberamp.helper_funcs import *


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
        self.effective_area_type = 'core_area'
        self.core_refractive_index = DEFAULT_GROUP_INDEX
        self.doping_profile = DopingProfile(ion_number_densities=[0], radii=[core_radius])
        self.default_signal_mode_shape_parameters = {'functional_form': 'bessel',
                                                     'mode_diameter': 0,
                                                     'overlaps': []}
        self.default_pump_mode_shape_parameters = {'functional_form': 'bessel',
                                                     'mode_diameter': 0,
                                                     'overlaps': []}

    def v_parameter(self, wl):
        return fiber_v_parameter(wl, self.core_radius, self.core_na)

    @property
    def num_ion_populations(self):
        return len(self.doping_profile.ion_number_densities)

    def nonlinear_effective_area(self, freq):
        """Returns the nonlinear effective area of the fundamental fiber mode with the given frequency. The method used
        is determined by the attribute self.effective_area_type.

        :param freq: The frequency of the optical signal (Hz).
        :type freq: float or numpy float array
        :returns: The nonlinear effective area
        :rtype: Same as argument type.

        """
        if self.effective_area_type == 'core_area':
            return self._effective_area_from_core_area(freq)
        elif self.effective_area_type == 'gaussian':
            return self._effective_area_from_gaussian_approximation(freq)
        elif self.effective_area_type == 'bessel':
            return self._effective_area_from_bessel_distribution(freq)

    def _effective_area_from_core_area(self, freq):
        """Returns a numpy array with len(freq) items. All items are equal to the fiber's core area."""
        return self.core_area() * np.ones_like(freq)

    def _effective_area_from_gaussian_approximation(self, freq):
        """Returns a numpy array with len(freq) items. The effective core area is calculated for each frequency as
        pi*(mfd/2)**2, where mfd is the mode field diameter. This is not exactly physically correct but computationally
        easy."""
        return effective_area_from_mfd(freq_to_wl(freq), self.core_radius, self.core_na)

    def _effective_area_from_bessel_distribution(self, freq):
        raise NotImplementedError()

    def core_area(self):
        """Returns the core area of the fiber defined as pi*r**2, where r is the core radius.

        :returns: Core area
        :rtype: float

        """
        return self.core_radius ** 2 * np.pi

    @abstractmethod
    def get_channel_emission_cross_section(self, freq, frequency_bandwidth):
        pass

    @abstractmethod
    def get_channel_absorption_cross_section(self, freq, frequency_bandwidth):
        pass

    @abstractmethod
    def saturation_parameter(self):
        pass

