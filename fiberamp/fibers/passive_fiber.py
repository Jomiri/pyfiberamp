from fiberamp.helper_funcs import *


class PassiveFiber:
    def __init__(self, length, core_r, background_loss, core_na):
        self.length = length
        self.core_r = core_r
        self.background_loss = background_loss
        self.core_na = core_na
        self.effective_area_type = 'core_area'

    def nonlinear_effective_area(self, freq):
        if self.effective_area_type == 'core_area':
            return self.effective_area_from_core_area(freq)
        elif self.effective_area_type == 'gaussian':
            return self.effective_area_from_gaussian_approximation(freq)
        elif self.effective_area_type == 'bessel':
            return self.effective_area_from_bessel_distribution(freq)

    def effective_area_from_core_area(self, freq):
        return self.core_area() * np.ones_like(freq)

    def core_area(self):
        return self.core_r**2 * np.pi

    def effective_area_from_gaussian_approximation(self, freq):
        return effective_area_from_mfd(freq_to_wl(freq), self.core_r, self.core_na)

    def effective_area_from_bessel_distribution(self, freq):
        raise NotImplementedError

    @staticmethod
    def eta():
        return 1

    @staticmethod
    def make_gain_spectrum(freq, mode_field_radii, slices):
        return 0

    @staticmethod
    def make_absorption_spectrum(freq, mode_field_radii, slices):
        return 0
