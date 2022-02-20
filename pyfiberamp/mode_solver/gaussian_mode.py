from scipy.integrate import quad
from pyfiberamp.helper_funcs import *
from pyfiberamp.mode_solver.mode_base import ModeBase


class GaussianMode(ModeBase):

    @classmethod
    def from_fiber_parameters(cls, a: float, core_na: float, wl: float):
        mode_radius = fundamental_mode_radius_petermann_2(wl, a, core_na)
        return GaussianMode(2*mode_radius)

    def __init__(self, mfd, core_radius):
        self.mode_radius = mfd / 2
        self.core_radius = core_radius

    def intensity(self, r: float, phi=None):
        return 2 / (np.pi * self.mode_radius**2) * np.exp(-2 * r**2 / self.mode_radius**2)

    @property
    def core_overlap(self):
        return self.radial_integral(0, self.core_radius) * self._angular_full_integral()

    def radial_integral(self, start: float, stop: float):
        return quad(lambda r: self.intensity(r) * r, start, stop, epsrel=1e-14)[0]

    def core_section_overlap(self, r_lim, phi_lim):
        assert r_lim[1] > r_lim[0] >= 0
        assert 2 * np.pi >= phi_lim[1] > phi_lim[0] >= 0
        return self.radial_integral(*r_lim) * (phi_lim[1]-phi_lim[0])

    def _angular_full_integral(self):
        return 2 * np.pi
