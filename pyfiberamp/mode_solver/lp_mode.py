import numpy as np
import inspect
from numpy.polynomial import Polynomial as Poly
from scipy.special import jv, kn
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pyfiberamp.mode_solver.lp_mode_solver
from pyfiberamp.helper_funcs import *
from pyfiberamp.mode_solver.mode_base import ModeBase


def n_fused_silica(wl: float):
    """Refractive index of fused silica from Sellmeier coefficients. From refractiveindex.info."""
    x = wl * 1e6
    return (1 + 0.6961663 / (1 - (0.0684043 / x) ** 2) + 0.4079426 / (1 - (0.1162414 / x) ** 2) + 0.8974794 / (
            1 - (9.896161 / x) ** 2)) ** .5


def beta(U: float, a: float, n_co: float, wl: float):
    """Propagation constant from mode solver's result U and fiber parameters."""
    return np.sqrt(n_co ** 2 * (2 * np.pi / wl) ** 2 - U ** 2 / a ** 2)


def core_index(na: float, wl: float):
    """Core refractive index assuming cladding is pure fused silica and na is constant over all wavelengths."""
    n_clad = n_fused_silica(wl)
    return np.sqrt(n_clad ** 2 + na ** 2)


class LPMode(ModeBase):
    def __init__(self, l: int, m: int, u: float, na: float, a: float, wl: float, rotation: str, cutoff_wl: float,
                 n_core_func=core_index):
        super().__init__(a)
        self.l = l
        self.m = m
        self.u = u
        self.na = na
        self.a = a
        self.wl = wl
        self.rotation = rotation
        self.cutoff_wl = cutoff_wl
        self.n_core = n_core_func
        assert rotation in ['sin', 'cos']
        self.norm_constant = 1 / self._normalization_integral()

    def __str__(self):
        return inspect.cleandoc(f"""
        {self.name} of 
        fiber with core diameter {2*self.a*1e6} um, NA {self.na} at {self.wl*1e9:.2f} nm
        Effective index: {self.effective_index:.6f}
        Core overlap: {self.core_overlap:.3f}
        Cutoff wavelength: {self.cutoff_wl*1e9:.2f} nm
        Effective area: {self.effective_area*1e12:.1f} um^2
        Effective MFD: {self.effective_mfd*1e6:.2f} um\n
        """)

    @property
    def name(self):
        rotation_str = f'{self.rotation} variant' if self.l != 0 else 'no variants'
        return f'LP_{self.l},{self.m} mode, {rotation_str}'

    @property
    def v(self):
        return fiber_v_parameter(self.wl, self.a, self.na)

    @property
    def omega(self):
        return wl_to_omega(self.wl)

    @property
    def propagation_constant(self):
        return beta(self.u, self.a, self.n_core(self.na, self.wl), self.wl)

    def dispersion(self, n_points_per_side=20, delta_wl=5e-12, deg=7):
        """Dispersion computation by the following technique:
        Run mode solver for a number of adjacent wavelengths (n_points_per_side*2) at delta_wl intervals.
        Compute propagation constant beta for each of those solutions.
        Compute angular frequency omega from the used wavelengths.
        Fit a polynomial to the data -> this is a function beta(omega)
        Differentiate said polynomial, eg. twice for beta_2 dispersion.
        Evaluate the derivative at the center wavelength.

        This is still an experimental feature. Please tune the parameters to make sure the solution converges.

        :param delta_wl: Wavelength step between fit points
        :type delta_wl: float
        :params
        """
        wl_points = np.linspace(self.wl - n_points_per_side * delta_wl,
                                self.wl + n_points_per_side * delta_wl,
                                2 * n_points_per_side + 1)
        solver = pyfiberamp.mode_solver.lp_mode_solver.LPModeSolver(self.l + 1, self.m + 1)
        betas = [solver.find_mode(self.l, self.m, self.a, self.na, wl).propagation_constant for wl in wl_points]
        omegas = wl_to_omega(wl_points)
        beta_fit = Poly.fit(omegas, betas, deg=deg)
        beta0 = beta_fit(self.omega)
        beta1 = beta_fit.deriv(1)(self.omega)
        beta2 = beta_fit.deriv(2)(self.omega)
        beta3 = beta_fit.deriv(3)(self.omega)
        return beta0, beta1, beta2, beta3

    @property
    def effective_index(self):
        """Computes the effective refractive index of the mode.

        :returns: effective refractive index
        :rtype: float
        """
        return self.propagation_constant / (2 * np.pi / self.wl)

    @property
    def group_index(self):
        """Computes the group index of the mode.

        :returns: group index
        :rtype: float
        """
        beta1 = self.dispersion()[1]
        return c * beta1

    @property
    def core_overlap(self):
        """Computes the mode's normalized overlap with the core."""
        return self.radial_integral(0, self.a) * self._angular_full_integral()

    @property
    def effective_mfd(self):
        return np.sqrt(self.effective_area / np.pi) * 2

    def radial_integral(self, start: float, stop: float):
        return (quad(lambda r: self._radial_intensity(r) * r, start, stop, epsrel=1e-14, points=self.a)[0]
                * self.norm_constant)

    def angular_integral(self, start: float, stop: float):
        return quad(self._angular_intensity, start, stop)[0]

    def intensity(self, r: float, phi: float):
        return self._radial_intensity(r) * self._angular_intensity(phi) * self.norm_constant

    def core_section_overlap(self, r_lim, phi_lim):
        assert r_lim[1] > r_lim[0] >= 0
        assert 2*np.pi >= phi_lim[1] > phi_lim[0] >= 0
        return self.radial_integral(*r_lim) * self.angular_integral(*phi_lim)

    def _angular_full_integral(self):
        return 2 * np.pi if self.l == 0 else np.pi

    def _normalization_integral(self):
        """Very important to give the core radius as points parameter to quad to get stable results."""
        radial_integral = quad(lambda r: self._radial_intensity(r) * r, 0, 10 * self.a, epsrel=1e-14, points=self.a)[0]
        return self._angular_full_integral() * radial_integral

    def _radial_intensity(self, r: float):
        """Un-normalized radial intensity of the mode"""
        w = np.sqrt(self.v ** 2 - self.u ** 2)
        jk = jv(self.l, self.u) / kn(self.l, w)
        radial_amplitude = (jv(self.l, self.u * r / self.a) * (r < self.a)
                            + jk * kn(self.l, w * r / self.a) * (r >= self.a))
        return radial_amplitude**2

    def _angular_intensity(self, phi: float):
        if self.rotation == 'sin':
            angular_amplitude = np.sin(self.l * phi)
        else:
            angular_amplitude = np.cos(self.l * phi)
        return angular_amplitude**2

    def plot_intensity(self):
        x = np.linspace(-2 * self.a, 2 * self.a, 1000)
        y = np.linspace(-2 * self.a, 2 * self.a, 1000)
        xv, yv = np.meshgrid(x, y)
        intensity = self.intensity(np.sqrt(xv ** 2 + yv ** 2), np.arctan2(yv, xv))
        xlim = np.array([x[0], x[-1]]) * 1e6
        ylim = xlim
        self._make_single_plot(intensity, xlim, ylim)
        plt.show()

    def _make_single_plot(self, intensity, xlim, ylim):
        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        plt.imshow(intensity, extent=(*xlim, *ylim))
        rotation_str = f', {self.rotation} variant' if self.l != 0 else ''
        plt.title(f'LP_{self.l}{self.m} mode{rotation_str}')
        plt.xlabel('X [um]')
        plt.ylabel('Y [um]')
        # Plot core as circle
        circ = plt.Circle((0, 0), radius=self.a*1e6, facecolor=(0, 0, 0, 0), edgecolor=(1, 1, 1, 0.5), linewidth=2)
        ax.add_patch(circ)