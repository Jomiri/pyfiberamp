from abc import ABC, abstractmethod
import numpy as np

from pyfiberamp.helper_funcs import *
from pyfiberamp.optical_channel import OpticalChannel


class FiberBase:
    """ FiberBase is the base class for the other fiber classes. It contains methods for calculating
    nonlinear effective area, mode field diameter and setting up optical channels. In principle, PassiveFiber class
    could act as a base class but subclassing ActiveFiber from PassiveFiber feels conceptually wrong."""

    @abstractmethod
    def __init__(self, length=0, core_radius=0, background_loss=0, core_na=0):
        """
        Parameters
        ----------
        length : float
            Fiber length
        core_radius : float
            Fiber core radius
        background_loss : float
            Linear loss (scattering etc.) in (1/m) NOT dB

        """
        self.length = length
        self.core_radius = core_radius
        self.background_loss = background_loss
        self.core_na = core_na
        self.effective_area_type = 'core_area'

    def nonlinear_effective_area(self, freq):
        """Returns the nonlinear effective area of the fundamental fiber mode with the given frequency. The method used
        is determined by the attribute self.effective_area_type.

        Parameters
        ----------
        freq : numpy float array
            The frequency of the optical signal (Hz).

        Returns
        -------
        nonlinear_effective_area : numpy float array"""

        if self.effective_area_type == 'core_area':
            return self.effective_area_from_core_area(freq)
        elif self.effective_area_type == 'gaussian':
            return self.effective_area_from_gaussian_approximation(freq)
        elif self.effective_area_type == 'bessel':
            return self.effective_area_from_bessel_distribution(freq)

    def effective_area_from_core_area(self, freq):
        """Returns a numpy array with len(freq) items. All items are equal to the fiber's core area."""
        return self.core_area() * np.ones_like(freq)

    def effective_area_from_gaussian_approximation(self, freq):
        """Returns a numpy array with len(freq) items. The effective core area is calculated for each frequency as
        pi*(mfd/2)**2, where mfd is the mode field diameter. This is not exactly physically correct but computationally
        easy."""
        return effective_area_from_mfd(freq_to_wl(freq), self.core_radius, self.core_na)

    def effective_area_from_bessel_distribution(self, freq):
        raise NotImplementedError()

    def core_area(self):
        """Returns the core area of the fiber defined as pi*r**2, where r is the core radius."""
        return self.core_radius ** 2 * np.pi

    def get_mode_field_diameter(self, freq, preset_mfd):
        """Returns the mode field diameter of the fiber. If mode field diameter was preset, returns the preset value
        instead.

        Parameters
        ----------
        freq : float or numpy float array
            The frequency of the optical signal (Hz).
        preset_mfd : float
            Possible user-provided value for the mode field diameter. Equals to 0 if automatic value should be used.

        Returns
        -------
        mode field diameter : float or numpy float array

        """

        if preset_mfd > 0:
            return preset_mfd
        else:
            return fundamental_mode_mfd_petermann_2(freq_to_wl(freq), self.core_radius, self.core_na)

    def create_in_core_single_frequency_channel(self, wl, power, preset_mfd, direction):
        """Returns an optical channel which describes the properties of a single frequency beam propagating
        in the fiber. Gain and absorption are calculated at this single frequency and the bandwidth is zero.

        Parameters
        ----------
        wl : float
           The wavelength of the beam
        power : float
           The input power of the beam
        preset_mfd : float
           User-defined mode field diameter of the beam. Equals to zero when automatic value is to be used.
        direction : int
            Beam propagation direction (+1 forward, -1 backward)

        Returns
        -------
           An OpticalChannel object
        """

        frequency = wl_to_freq(wl)
        frequency_bandwidth = 0
        mfd = self.get_mode_field_diameter(frequency, preset_mfd)
        mode_field_radius = mfd / 2
        gain = self.get_channel_gain(frequency, mode_field_radius)
        absorption = self.get_channel_absorption(frequency, mode_field_radius)
        loss = self.background_loss
        return OpticalChannel(frequency, frequency_bandwidth, power, direction, mfd, gain, absorption, loss)

    def create_in_core_finite_bandwidth_channel(self, wl, wl_bandwidth, power, preset_mfd, direction):
        """Returns an optical channel which describes the properties of a finite bandwidth beam propagating
        in the fiber. Gain and absorption are calculated at and average over the starting, middle and end points
        the bandwidth and the bandwidth has a non-zero value.

        Parameters
        ----------
        wl : float
            The wavelength of the beam
        wl_bandwidth : float
            The wavelength bandwidth of the beam.
        power : float
            The input power of the beam
        preset_mfd : float
            User-defined mode field diameter of the beam. Equals to zero when automatic value is to be used.
        direction : int
            Beam propagation direction (+1 forward, -1 backward)

        Returns
        -------
           An OpticalChannel object
        """

        center_frequency = wl_to_freq(wl)
        frequency_bandwidth = wl_bw_to_freq_bw(wl_bandwidth, wl)
        mfd = self.get_mode_field_diameter(center_frequency, preset_mfd)
        mode_field_radius = mfd / 2
        gain = self.finite_bandwidth_gain(center_frequency, frequency_bandwidth, mode_field_radius)
        absorption = self.finite_bandwidth_absorption(center_frequency, frequency_bandwidth, mode_field_radius)
        loss = self.background_loss
        return OpticalChannel(center_frequency, frequency_bandwidth, power, direction, mfd, gain, absorption, loss)

    def create_in_core_forward_single_frequency_channel(self, wl, power, preset_mfd):
        """Wrapper function to hide the direction parameter from outer classes."""
        return self.create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=+1)

    def create_in_core_backward_single_frequency_channel(self, wl, power, preset_mfd):
        """Wrapper function to hide the direction parameter from outer classes."""
        return self.create_in_core_single_frequency_channel(wl, power, preset_mfd, direction=-1)

    def create_in_core_forward_finite_bandwidth_channel(self, wl, wl_bandwidth, power, preset_mfd):
        """Wrapper function to hide the direction parameter from outer classes."""
        return self.create_in_core_finite_bandwidth_channel(wl, wl_bandwidth, power, preset_mfd, direction=+1)

    def create_in_core_backward_finite_bandwidth_channel(self, wl, wl_bandwidth, power, preset_mfd):
        """Wrapper function to hide the direction parameter from outer classes."""
        return self.create_in_core_finite_bandwidth_channel(wl, wl_bandwidth, power, preset_mfd, direction=-1)


    @abstractmethod
    def get_channel_gain(self, freq, mode_field_radius):
        pass

    @abstractmethod
    def get_channel_absorption(self, freq, mode_field_radius):
        pass

    @abstractmethod
    def finite_bandwidth_gain(self, center_frequency, frequency_bandwidth, mode_field_radius):
        pass

    @abstractmethod
    def finite_bandwidth_absorption(self, center_frequency, frequency_bandwidth, mode_field_radius):
        pass

    @abstractmethod
    def saturation_parameter(self):
        pass

    def is_passive_fiber(self):
        return self.saturation_parameter() == 1
