import numpy as np
import warnings
from scipy.special import jv, kn
from scipy.optimize import brentq
import scipy.special as sc
from collections import namedtuple
import pyfiberamp.mode_solver.lp_mode
from pyfiberamp.helper_funcs import *


CutoffLimits = namedtuple('CutoffLimits', 'l m lower upper')
ModeSolverResult = namedtuple('Result', 'root v_cutoff converged')


class ModeNotFoundError(RuntimeError):
    pass


class LPModeSolver:
    """
    This is a mode solver class for finding linearly polarized (LP) modes of a cylindrical step-index fiber.
    During initialization, the mode solver computes cutoffs for each mode up to given maximum order.
    """
    def __init__(self, l_max: int, m_max: int):
        """
        Constructor for the LP mode solver.
        :param l_max: The highest l number the solver is initialized to find.
        :param m_max: The highest m number the solver is initialized to find.
        """
        self.l_max = l_max
        self.m_max = m_max
        self.cutoff_table = self._create_cutoff_table(l_max, m_max)
        self.upper_bracket_shift = 1e-6

    def __str__(self):
        return f"""LPModeSolver initialized with l_max={self.l_max}, m_max={self.m_max}."""

    def check_lm(self, l: int, m: int):
        if l > self.l_max:
            raise ModeNotFoundError(f'Target l = {l} exceeds l_max = {self.l_max}')
        if m > self.m_max:
            raise ModeNotFoundError(f'Target m = {m} exceeds m_max = {self.m_max}')

    @staticmethod
    def cutoff_wl(v_cutoff, core_radius, na):
        if v_cutoff == 0:
            return 0
        return 2*np.pi / v_cutoff * core_radius * na

    @staticmethod
    def _create_cutoff_table(l_max: int, m_max: int):
        """We compute and save the lower and upper cuttoffs for each mode beforehand for efficiency reasons.
        l_max and m_max must be chosen high enough that all modes fit within them."""
        zeros = np.vstack(tuple(sc.jn_zeros(l, m_max + 3) for l in range(l_max + 1)))
        cutoff_table = [CutoffLimits(l=0, m=1, lower=0, upper=sc.jn_zeros(0, 1)[0])]
        for l in range(l_max + 1):
            n = l - 1 if l > 0 else 1
            for m in range(m_max):
                m_table = m + 1 + (l == 0)
                if m_table > m_max:
                    break
                cutoff_table.append(CutoffLimits(l=l,
                                                 m=m_table,
                                                 lower=zeros[n, m],
                                                 upper=zeros[l, m_table-1]))
        return sorted(cutoff_table, key=lambda x: x.lower)

    def _bracket_for_mode(self, l: int, m: int, v: float):
        _, _, lower, upper = next(x for x in self.cutoff_table if (x[0] == l and x[1] == m))
        upper = min(v, upper) - self.upper_bracket_shift
        return lower, upper

    def solve_for_u(self, l: int, m: int, v: float):
        self.check_lm(l, m)
        start, stop = self._bracket_for_mode(l, m, v)
        if start >= stop:
            return ModeSolverResult(root=0, v_cutoff=None, converged=False)
        else:
            def mode_func(u: float):
                w = np.sqrt(v ** 2 - u ** 2)
                return u * jv(l + 1, u) / jv(l, u) - w * kn(l + 1, w) / kn(l, w)
            if mode_func(start) * mode_func(stop) > 0: #Mode very close to cutoff -> cannot be solved
                return ModeSolverResult(root=0, v_cutoff=None, converged=False)
            root, res = brentq(mode_func, a=start, b=stop, full_output=True)
            return ModeSolverResult(root=root, v_cutoff=start, converged=res.converged)

    def find_mode(self, l: int, m: int, core_radius: float, na: float, wl: float, rotation='cos'):
        """
        Run LP mode solver in order to find a single specific mode.
        :param l: The mode's l number
        :param m: The mode's m number
        :param core_radius: The fiber's core radius
        :param na: The fiber's core numerical aperture
        :param wl: The mode's wavelength
        :param rotation: The rotational variant of the mode (sin or cos) when applicable.
        :returns Instance of LPMode class if the mode is found
        :raises ModeNotFoundError if the mode is not found
        """
        if rotation not in ['sin', 'cos']:
            raise ModeNotFoundError('Rotation must be sin or cos.')
        if l == 0 and rotation == 'sin':
            raise ModeNotFoundError('l = 0 modes have no rotational variants.')
        v = fiber_v_parameter(wl, core_radius, na)
        res = self.solve_for_u(l, m, v)
        if not res.converged:
            raise ModeNotFoundError(f'LP mode l={l}, m={m} not found')
        if l == 0 and m == 1:
            cutoff_wl = np.inf
        else:
            cutoff_wl = self.cutoff_wl(res.v_cutoff, core_radius, na)
        return pyfiberamp.mode_solver.lp_mode.LPMode(l, m, res.root, na, core_radius, wl, rotation, cutoff_wl)

    def find_all_modes(self, core_radius: float, na: float, wl: float):
        """
        Solve for a list of all modes with given step-index fiber parameters
        :param core_radius: The fiber's core radius
        :param na: The fiber's core numerical aperture
        :param wl: The mode's wavelength
        :return: List of all found LPModes
        """
        v = fiber_v_parameter(wl, core_radius, na)
        modes_out = []
        for mode_params in self.cutoff_table:
            try:
                found_mode_cos = self.find_mode(mode_params.l, mode_params.m, core_radius, na, wl,'cos')
                modes_out.append(found_mode_cos)
                if mode_params.l > 0:
                    found_mode_sin = self.find_mode(mode_params.l, mode_params.m, core_radius, na, wl, 'sin')
                    modes_out.append(found_mode_sin)
                if found_mode_cos.l == self.l_max or found_mode_cos.m == self.m_max:
                    warnings.warn('Found mode with l=l_max or m=m_max. Possible higher order modes could be missed.')
            except ModeNotFoundError:
                break
        return modes_out


# Pre-initialized mode solver
default_mode_solver = LPModeSolver(l_max=100, m_max=50)


if __name__ == '__main__':
    mode_solver = LPModeSolver(l_max=10, m_max=5)
    all_modes = mode_solver.find_all_modes(core_radius=10e-6,
                                           na=0.065,
                                           wl=1064e-9)

    for mode in all_modes:
        mode.plot_intensity()
        print(mode)

    #all_modes = mode_solver.find_all_modes(core_radius=10e-6,
    #                                       na=0.065,
    #                                       freq=1064e-9)

