from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import quad


class ModeBase(ABC):
    """
    Base class for different fiber mode classes
    """

    def __init__(self, core_radius):
        self.core_radius = core_radius

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

    @abstractmethod
    def _radial_intensity(self, r):
        pass

    @property
    def effective_area(self):
        r_end = 10 * self.core_radius
        upper_func = lambda r: self._angular_full_integral() * r * self._radial_intensity(r)
        lower_func = lambda r: self._angular_full_integral() * r * self._radial_intensity(r) ** 2
        upper = quad(upper_func, 0, r_end, points=[self.core_radius])[0]
        lower = quad(lower_func, 0, r_end, points=[self.core_radius])[0]
        return upper ** 2 / lower
