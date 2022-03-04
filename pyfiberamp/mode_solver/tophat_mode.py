import numpy as np
from pyfiberamp.mode_solver.mode_base import ModeBase


class TophatMode(ModeBase):
    """
    The TophatMode class represents a fiber mode with constant intensity in a disk-like region and instant fall-off
    outside it.
    """
    def __init__(self, mode_radius: float, core_radius: float):
        """
        Constructor

        :param mode_radius: Radius of the top-hat "disk"
        :param core_radius: Radius of the fiber core
        """
        super().__init__(core_radius)
        self.mode_radius = mode_radius
        self.normalization_const = 1 / (np.pi * mode_radius ** 2)

    def intensity(self, r, phi=None):
        """
        Mode intensity at polar coordinates (r, phi). The angle is redundant since TophatMode is radially symmetric.

        :param r: Radial distance from the center of the fiber
        :param phi: Angle
        :return: Normalized intensity I(r, phi)

        """
        return self._radial_intensity(r)

    def _radial_intensity(self, r):
        return (r <= self.mode_radius) * self.normalization_const

    @property
    def core_overlap(self):
        """
        The mode's total overlap with the fiber core.

        :return: Overlap as 0...1
        """
        return self.radial_integral(0, self.core_radius)

    def radial_integral(self, start: float, stop: float):
        if stop > self.mode_radius:
            stop = self.mode_radius
        if start > self.mode_radius:
            start = self.mode_radius
        return 0.5 * self.normalization_const * (stop**2 - start**2)

    def core_section_overlap(self, r_lim, phi_lim):
        """
        The mode's overlap with a single core section defined by r_lim, phi_lim

        :param r_lim: Numpy array containing min and max radius defining the section.
        :param phi_lim: Numpy array containing min and max angles defining the section.
        :return: Overlap with the core section 0...1
        """
        assert r_lim[1] > r_lim[0] >= 0
        assert 2 * np.pi >= phi_lim[1] > phi_lim[0] >= 0
        return self.radial_integral(*r_lim) * (phi_lim[1] - phi_lim[0])

    def _angular_full_integral(self):
        return 2 * np.pi

