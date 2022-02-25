import numpy as np
from pyfiberamp.mode_solver.mode_base import ModeBase


class TophatMode(ModeBase):

    def __init__(self, mode_radius, core_radius):
        super().__init__(core_radius)
        self.mode_radius = mode_radius
        self.normalization_const = 1 / (np.pi * mode_radius ** 2)

    def intensity(self, r, phi=None):
        return self._radial_intensity(r)

    def _radial_intensity(self, r):
        return (r <= self.mode_radius) * self.normalization_const

    @property
    def core_overlap(self):
        return self.radial_integral(0, self.core_radius)

    def radial_integral(self, start: float, stop: float):
        if stop > self.mode_radius:
            stop = self.mode_radius
        if start > self.mode_radius:
            start = self.mode_radius
        return 0.5 * self.normalization_const * (stop**2 - start**2)

    def core_section_overlap(self, r_lim, phi_lim):
        assert r_lim[1] > r_lim[0] >= 0
        assert 2 * np.pi >= phi_lim[1] > phi_lim[0] >= 0
        return self.radial_integral(*r_lim) * (phi_lim[1] - phi_lim[0])

    def _angular_full_integral(self):
        return 2 * np.pi

