from pyfiberamp.simulation_result import SimulationResult
from scipy.integrate import solve_bvp

from pyfiberamp.channels import Channels
from pyfiberamp.helper_funcs import *
from pyfiberamp.steady_state.initial_guess import InitialGuessFromParameters, InitialGuessFromArray
from pyfiberamp.steady_state.models.giles_model import GilesModel
from pyfiberamp.steady_state.steady_state_boundary_conditions import BasicBoundaryConditions


class SteadyStateSimulation:
    """
    SteadyStateSimulation is the main class used for running steady state Giles model simulations \
    without Raman scattering. Only one ion population is supported. The class defines the fiber, boundary conditions and
    optical channels used in the simulation.
    """

    def __init__(self):
        self.fiber = None
        self.model = GilesModel
        self.boundary_conditions = BasicBoundaryConditions
        self.initial_guess = InitialGuessFromParameters()
        self.channels = Channels()
        self.solver_verbosity = 2

    def add_cw_signal(self, wl, power, wl_bandwidth=0, mode_shape_parameters=None, label=''):
        """Adds a new forward propagating single-frequency CW signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> one of ['bessel', 'gaussian', 'tophat']  \
         *mode_diameter* -> float \
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel
        :type label: str

        """
        self.channels.add_forward_signal(wl, wl_bandwidth, power, mode_shape_parameters, label)

    def add_forward_pump(self, wl, power, wl_bandwidth=0, mode_shape_parameters=None, label=''):
        """Adds a new forward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> one of ['bessel', 'gaussian', 'tophat']  \
         *mode_diameter* -> float \
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel
        :type label: str

        """
        self.channels.add_forward_pump(wl, wl_bandwidth, power, mode_shape_parameters, label)

    def add_backward_pump(self, wl, power, wl_bandwidth=0, mode_shape_parameters=None, label=''):
        """Adds a new backward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> one of ['bessel', 'gaussian', 'tophat']  \
         *mode_diameter* -> float \
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel
        :type label: str

        """
        self.channels.add_backward_pump(wl, wl_bandwidth, power, mode_shape_parameters, label)

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

    def run(self, tol=1e-3):
        """Runs the simulation, i.e. calculates the steady state of the defined fiber amplifier. ASE or raman
        simulations might require higher tolerance than the default value.
        It is best to decrease the tolerance until the result no longer changes.

        :param tol: Target error tolerance of the solver.
        :type tol: float

        """
        if self.fiber.num_ion_populations > 1:
            raise RuntimeError('Use DynamicSimulation for calculations with multiple transverse ion populations.')
        self.channels.set_fiber(self.fiber)
        boundary_condition_residual = self.boundary_conditions(self.channels)
        model = self.model(self.channels, self.fiber)
        rate_equation_rhs, upper_level_func = model.make_rate_equation_rhs()

        self.initial_guess.initialize(self.channels.get_input_powers())
        guess = self.initial_guess.as_array()
        sol = solve_bvp(rate_equation_rhs, boundary_condition_residual,
                        self._start_z(), guess, max_nodes=SOLVER_MAX_NODES, tol=tol, verbose=self.solver_verbosity)
        assert sol.success, 'Error: The solver did not converge.'
        return self._finalize(sol, upper_level_func)

    def set_guess_parameters(self, guess_parameters):
        """Overrides the default initial guess parameters.

        :param guess_parameters: Parameters used to create the initial guess array
        :type guess_parameters: Instance of GuessParameters class

        :Example:

        from pyfiberamp import GuessParameters, GainShapes
        params = GuessParameters()
        params.signal.set_gain_shape(GainShapes.LINEAR)
        params.pump.set_gain_db(-20)
        simulation.set_guess_parameters(params)

        """

        self.initial_guess.params = guess_parameters

    def set_guess_array(self, array, force_node_number=None):
        """Use an existing array as the initial guess. Typically this array is the result of a previous simulation
        with sligthly different parameters. Note that the number of simulated beams/channels must be the same.

        :param array: The initial guess array
        :type array: numpy array
        :param force_node_number: The new number of columns in the resized array.
        :type force_node_number: int, optional

        """

        self.initial_guess = InitialGuessFromArray(array, force_node_number)

    def set_number_of_nodes(self, N):
        """Override the default number of nodes used by the solver. The solver will increase the number of nodes if
         necessary.

         :param N: New starting number of nodes used by the solver.
         :type N: int

         """
        self.initial_guess.npoints = N

    def _start_z(self):
        """Creates the linear starting grid."""
        return np.linspace(0, self.fiber.length, self.initial_guess.npoints)

    def _finalize(self, sol, upper_level_func):
        """Creates the SimulationResult object from the solution object."""
        res = SimulationResult(z=sol.x,
                               powers=sol.y,
                               upper_level_fraction=upper_level_func(sol.y),
                               channels=self.channels,
                               fiber=self.fiber)
        return res




