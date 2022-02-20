from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import quad


class ModeBase(ABC):
    """
    Base class for different fiber mode classes
    """

    @abstractmethod
    def core_overlap(self):
        pass

    @abstractmethod
    def intensity(self, r, phi):
        pass

    @abstractmethod
    def _angular_full_integral(self):
        pass

    @abstractmethod
    def core_section_overlap(self, r_lim, phi_lim):
        pass

    def nonlinear_effective_area(self, core_radius):
        r_end = 10 * core_radius
        upper_func = lambda r: self._angular_full_integral() * r * self.intensity(r, 0)
        lower_func = lambda r: self._angular_full_integral() * r * self.intensity(r, 0) ** 2
        upper = quad(upper_func, 0, r_end, points=[core_radius])[0]
        lower = quad(lower_func, 0, r_end, points=[core_radius])[0]
        return upper ** 2 / lower
