from scipy.integrate import solve_bvp

from fiberamp.models.giles_model import GilesModel
from .boundary_conditions import BasicBoundaryConditions
from .helper_funcs import *
from .initial_guess_maker import InitialGuessMaker
from .channels import Channels
from .simulation_result import SimulationResult


class FiberAmplifierSimulation:
    def __init__(self, fiber):
        self.fiber = fiber
        self.model = GilesModel
        self.channels = Channels(fiber)
        self.slices = {}

    def add_cw_signal(self, wl, power, mode_field_diameter=0.0):
        self.channels.add_forward_signal(wl, power, mode_field_diameter)

    def add_pulsed_signal(self, wl, power, f_rep, fwhm_duration, mode_field_diameter=0.0):
        self.channels.add_pulsed_forward_signal(wl, power, f_rep, fwhm_duration, mode_field_diameter)

    def add_co_pump(self, wl, power, mode_field_diameter=0.0):
        self.channels.add_forward_pump(wl, power, mode_field_diameter)

    def add_counter_pump(self, wl, power, mode_field_diameter=0.0):
        self.channels.add_backward_pump(wl, power, mode_field_diameter)

    def add_ase(self, wl_start, wl_end, n_bins):
        self.channels.add_ase(wl_start, wl_end, n_bins)

    def run(self, npoints, tol=1e-3):
        self._init_slices()

        boundary_condition_residual = BasicBoundaryConditions(self.channels)
        model = self.model(self.channels, self.fiber)
        rate_equation_rhs, upper_level_func = model.make_rate_equation_rhs()

        guess_maker = InitialGuessMaker(self.channels.get_input_powers(), self.slices, self._start_z(npoints))
        guess = guess_maker.make_guess()
        sol = solve_bvp(rate_equation_rhs, boundary_condition_residual,
                        self._start_z(npoints), guess, max_nodes=SOLVER_MAX_NODES, tol=tol, verbose=2)
        return self._finalize(sol, upper_level_func)

    def _start_z(self, npoints):
        return np.linspace(0, self.fiber.length, npoints)

    def _finalize(self, sol, upper_level_func):
        res = SimulationResult(sol)
        res.upper_level_fraction = upper_level_func(sol.y)
        res = self._add_wls_and_slices_to_result(res)
        return res

    def _init_slices(self):
        self.slices = self.channels.get_slices()

    def _add_wls_and_slices_to_result(self, res):
        res.slices = self.slices
        res.wavelengths = self.channels.get_wavelengths()
        return res




