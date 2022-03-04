from scipy.integrate import quad
from pyfiberamp.helper_funcs import *
from pyfiberamp.mode_solver.mode_base import ModeBase


class GaussianMode(ModeBase):
    """
    The GaussianMode class describes a fundamental mode in an optical fiber using Gaussian approximation for the
    mode shape.
    """
    @classmethod
    def from_fiber_parameters(cls, core_radius: float, na: float, wl: float):
        """
        Initialize a Gaussian mode from fiber parameters.

        :param core_radius: Fiber core radius
        :param na: Fiber core numerical aperture
        :param wl: Wavelength of the mode
        :return:
        """
        mode_radius = fundamental_mode_radius_petermann_2(wl, core_radius, na)
        return GaussianMode(2*mode_radius, core_radius)

    def __init__(self, mfd, core_radius):
        super().__init__(core_radius)
        self.mode_radius = mfd / 2

    def intensity(self, r: float, phi=None):
        """
        Normalized mode intensity at radial location r.

        :param r: Radial coordinate away from the center of the fiber core
        :param phi: Angular coordinate, not needed here but included for compatability with LPMode class
        :return: Intensity
        """
        return self._radial_intensity(r)

    def _radial_intensity(self, r):
        return 2 / (np.pi * self.mode_radius**2) * np.exp(-2 * r**2 / self.mode_radius**2)

    @property
    def core_overlap(self):
        """
        The mode's total overlap with the fiber core.

        :return: Overlap as 0...1
        """
        return self.radial_integral(0, self.core_radius) * self._angular_full_integral()

    def radial_integral(self, start: float, stop: float):
        return quad(lambda r: self.intensity(r) * r, start, stop, epsrel=1e-14)[0]

    def core_section_overlap(self, r_lim, phi_lim):
        """
        The mode's overlap with a single core section defined by r_lim, phi_lim

        :param r_lim: Numpy array containing min and max radius defining the section.
        :param phi_lim: Numpy array containing min and max angles defining the section.
        :return: Overlap with the core section 0...1
        """
        assert r_lim[1] > r_lim[0] >= 0
        assert 2 * np.pi >= phi_lim[1] > phi_lim[0] >= 0
        return self.radial_integral(*r_lim) * (phi_lim[1]-phi_lim[0])

    def _angular_full_integral(self):
        return 2 * np.pi
