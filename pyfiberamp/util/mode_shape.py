import numpy as np
from scipy.special import j0 as J0
from scipy.special import k0 as K0
from scipy.special import j1 as J1
from scipy.special import k1 as K1
from scipy.integrate import quad
from scipy.optimize import brenth
from pyfiberamp.helper_funcs import *
import warnings


class ModeShape:
    def __init__(self, fiber, wavelength, mode_shape_parameters):
        functional_form = mode_shape_parameters['functional_form']
        radius = mode_shape_parameters['mode_diameter'] / 2

        self.mode_func = None
        if functional_form == 'bessel':
            if radius > 0:
                warnings.warn('Bessel mode cannot use predefined mode diameter.')
            self.mode_func = self.solve_fundamental_bessel_mode(fiber, wavelength)
        elif functional_form == 'gaussian':
            if radius == 0:
                self.mode_func = self.solve_fundamental_gaussian_mode(fiber, wavelength)
            else:
                self.mode_func = self.make_normalized_gaussian_mode(radius)
        elif functional_form == 'tophat':
            if radius == 0:
                radius = fiber.core_radius
            self.mode_func = self.make_normalized_top_hat_mode(radius)
        else:
            raise RuntimeError('Unsupported mode shape: {}.'.format(functional_form))

    def solve_fundamental_bessel_mode(self, fiber, wavelength):
        a = fiber.core_radius
        na_core = fiber.core_na
        n_core = fiber.core_refractive_index
        n_clad = np.sqrt(n_core**2 - na_core**2)
        k0 = 2 * np.pi / wavelength
        k_core = k0 * n_core
        k_clad = k0 * n_clad

        mode_matching_func = self.make_bessel_mode_matching_func(k_core, k_clad, a)
        beta = self.solve_propagation_constant(mode_matching_func, k_core, k_clad)
        return self.make_normalized_bessel_mode_func(beta, k_core, k_clad, a)

    def solve_fundamental_gaussian_mode(self, fiber, wavelength):
        a = fiber.core_radius
        na_core = fiber.core_na
        mode_radius = fundamental_mode_radius_petermann_2(wavelength, a, na_core)
        return self.make_normalized_gaussian_mode(mode_radius)

    def get_ring_overlaps(self, radii):
        radii_with_zero = np.hstack((0, radii))
        start_radii = radii_with_zero[:-1]
        end_radii = radii_with_zero[1:]
        overlaps = np.zeros(len(radii))
        for i in range(len(radii)):
            overlaps[i] = self._ring_overlap_integral(start_radii[i], end_radii[i])
        return overlaps

    def _ring_overlap_integral(self, r_start, r_end):
        integrand_func = lambda r: 2 * np.pi * r * self.mode_func(r)
        overlap, _ = quad(integrand_func, r_start, r_end)
        return overlap

    @staticmethod
    def make_normalized_top_hat_mode(mode_radius):
        normalization_const = np.pi * mode_radius**2

        def f(r):
            if r <= mode_radius:
                return 1 / normalization_const
            else:
                return 0

        return f

    @staticmethod
    def make_normalized_gaussian_mode(mode_radius):
        def f(r):
            return 1 / (np.pi * mode_radius**2) * np.exp(-r**2 / mode_radius**2)
        return f

    @staticmethod
    def make_normalized_bessel_mode_func(beta, k_core, k_clad, a):
        u = a * np.sqrt(k_core ** 2 - beta ** 2)
        v = a * np.sqrt(beta ** 2 - k_clad ** 2)
        V = np.sqrt(u ** 2 + v ** 2)

        def f(r):
            if r < a:
                return 1 / np.pi * (v / (a * V) * J0(u / a * r) / J1(u)) ** 2
            else:
                return 1 / np.pi * (u / (a * V) * K0(v / a * r) / K1(v)) ** 2
        return f

    @staticmethod
    def solve_propagation_constant(mode_matching_func, k_core, k_clad):
        eps = 1e-6
        beta_lower_bound = k_clad + eps
        beta_upper_bound = k_core - eps
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda r: abs(mode_matching_func(r)), bounds=[beta_lower_bound, beta_upper_bound],
                               method='bounded')
        beta = res.x
        assert res.success
        #beta, res_data = brenth(mode_matching_func, beta_lower_bound, beta_upper_bound, full_output=True)
        #assert res_data.converged
        return beta

    @staticmethod
    def make_bessel_mode_matching_func(k_core, k_clad, core_radius):
        def f(beta):
            u = core_radius * np.sqrt(k_core ** 2 - beta ** 2)
            v = core_radius * np.sqrt(beta ** 2 - k_clad ** 2)
            return v * J0(u) / J1(u) - u * K0(v) / K1(v)
        return f
