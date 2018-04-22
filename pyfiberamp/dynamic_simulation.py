
from pyfiberamp.channels import Channels
from pyfiberamp.helper_funcs import *
from pyfiberamp.initial_guess import InitialGuessFromParameters, InitialGuessFromArray
from pyfiberamp.simulation_result import SimulationResult
from pyfiberamp.models import DynamicModel
from pyfiberamp.finite_difference_solver import FiniteDifferenceSolver
from collections import namedtuple


class DynamicSimulation:
    """FiberAmplifierSimulation is the main class used for running Giles model simulations without Raman scattering.
    The class defines the fiber, boundary conditions and optical channels used in the simulation."""
    def __init__(self):
        """

        :param fiber: The fiber used in the simulation.
        :type fiber: class instance derived from FiberBase

        """
        self.fiber = None
        self.model = DynamicModel
        self.channels = Channels()
        self.npoints = START_NODES

    def add_cw_signal(self, wl, power, mode_field_diameter=0.0):
        """Adds a new forward propagating single-frequency CW signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param mode_field_diameter: Mode field diameter of the signal.
         If left undefined, will be calculated using the Petermann II equation.
        :type mode_field_diameter: float, optional

        """
        self.channels.add_forward_signal(wl, power, mode_field_diameter)

    def add_forward_pump(self, wl, power, mode_field_diameter=0.0):
        """Adds a new forward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param mode_field_diameter: Mode field diameter of the signal.
         If left undefined, will be calculated using the Petermann II equation.
        :type mode_field_diameter: float, optional
        """
        self.channels.add_forward_pump(wl, power, mode_field_diameter)

    def add_backward_pump(self, wl, power, mode_field_diameter=0.0):
        """Adds a new backward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param mode_field_diameter: Mode field diameter of the signal.
         If left undefined, will be calculated using the Petermann II equation.
        :type mode_field_diameter: float, optional

        """
        self.channels.add_backward_pump(wl, power, mode_field_diameter)

    def add_ase(self, wl_start, wl_end, n_bins):
        """Adds amplified spontaneous emission (ASE) channels.
        Using more channels improves accuracy, but incurs a heavier computational cost to the simulation.

        :param wl_start: The shorted wavelength of the ASE band
        :type wl_start: float
        :param wl_end: The longest wavelength of the ASE band
        :type wl_end: float
        :param n_bins: The number of simulated ASE channels.
        :type n_bins: float

        """
        self.channels.add_ase(wl_start, wl_end, n_bins)

    def run(self, z_nodes, max_times, propagation_speeds, tol):
        """Runs the simulation, i.e. calculates the steady state of the defined fiber amplifier. ASE or raman
        simulations might require higher tolerance than the default value.
        It is best to decrease the tolerance until the result no longer changes.

        :param tol: Target error tolerance of the solver.
        :type tol: float

        """
        self.channels.set_fiber(self.fiber)
        solver = FiniteDifferenceSolver(channels=self.channels, fiber=self.fiber)
        P, N2, z = solver.multi_step_steady_state_simulation(z_nodes, max_times, propagation_speeds, tol)
        solution = namedtuple('Solution', ('x', 'y'))
        sol = solution(x=z, y=P)
        return self._finalize(sol, N2 / self.fiber.ion_number_density)

    def _finalize(self, sol, upper_level_fraction):
        """Creates the SimulationResult object from the solution object."""
        res = SimulationResult(solution=sol,
                               upper_level_fraction=upper_level_fraction,
                               slices=self.channels.get_slices(),
                               wavelengths=self.channels.get_wavelengths(),
                               is_passive_fiber=self.fiber.is_passive_fiber())
        return res



