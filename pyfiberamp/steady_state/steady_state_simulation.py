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

    def __init__(self, fiber):
        self.fiber = fiber
        self.model = GilesModel
        self.boundary_conditions = BasicBoundaryConditions
        self.initial_guess = InitialGuessFromParameters()
        self.channels = Channels(fiber)
        self.solver_verbosity = 2

    def add_forward_signal(self, wl: float, input_power: float, wl_bandwidth=0.0, loss=None, mode=None, channel_id=None,
                           reflection_target_id=None, reflectance=0.0):
        """Adds a new forward propagating single-frequency CW signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means seeding by
         spontaneous emission.
        :type wl_bandwidth: float
        :param loss: Background loss of the channel. If None, the fiber's default loss is used.
        :type loss: float
        :param mode: Fiber mode class defining the channel's mode shape.
        :type mode: Subclass of ModeBase (such as LPMode or TophatMode)
        :param channel_id: Identifier for the channel, used for reflection definitions and plotting
        :type channel_id: int or str
        :param reflection_target_id: Identifier for the target channel that this channel reflects to
        :type reflection_target_id: int or str
        :param reflectance: Reflectance at the end of the channel 0<=R<=1
        :type reflectance: float

        """
        self.channels.create_channel(channel_type='signal',
                                     direction=1,
                                     fiber=self.fiber,
                                     input_power=input_power,
                                     wl=wl,
                                     mode=mode,
                                     channel_id=channel_id,
                                     wl_bandwidth=wl_bandwidth,
                                     loss=loss,
                                     reflection_target_id=reflection_target_id,
                                     reflectance=reflectance)

    def add_backward_signal(self, wl: float, input_power: float, wl_bandwidth=0.0, loss=None,
                            mode=None, channel_id=None,
                            reflection_target_id=None, reflectance=0.0):
        """Adds a new forward propagating single-frequency CW signal to the simulation.

         :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means seeding by
         spontaneous emission.
        :type wl_bandwidth: float
        :param loss: Background loss of the channel. If None, the fiber's default loss is used.
        :type loss: float
        :param mode: Fiber mode class defining the channel's mode shape.
        :type mode: Subclass of ModeBase (such as LPMode or TophatMode)
        :param channel_id: Identifier for the channel, used for reflection definitions and plotting
        :type channel_id: int or str
        :param reflection_target_id: Identifier for the target channel that this channel reflects to
        :type reflection_target_id: int or str
        :param reflectance: Reflectance at the end of the channel 0<=R<=1
        :type reflectance: float

        """
        self.channels.create_channel(channel_type='signal',
                                     direction=-1,
                                     fiber=self.fiber,
                                     input_power=input_power,
                                     wl=wl,
                                     mode=mode,
                                     channel_id=channel_id,
                                     wl_bandwidth=wl_bandwidth,
                                     loss=loss,
                                     reflection_target_id=reflection_target_id,
                                     reflectance=reflectance)

    def add_forward_pump(self, wl: float, input_power: float, wl_bandwidth=0.0, loss=None, mode=None, channel_id=None,
                         reflection_target_id=None, reflectance=0.0):
        """Adds a new forward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means seeding by
         spontaneous emission.
        :type wl_bandwidth: float
        :param loss: Background loss of the channel. If None, the fiber's default loss is used.
        :type loss: float
        :param mode: Fiber mode class defining the channel's mode shape.
        :type mode: Subclass of ModeBase (such as LPMode or TophatMode)
        :param channel_id: Identifier for the channel, used for reflection definitions and plotting
        :type channel_id: int or str
        :param reflection_target_id: Identifier for the target channel that this channel reflects to
        :type reflection_target_id: int or str
        :param reflectance: Reflectance at the end of the channel 0<=R<=1
        :type reflectance: float

        """
        self.channels.create_channel(channel_type='pump',
                                     direction=1,
                                     fiber=self.fiber,
                                     input_power=input_power,
                                     wl=wl,
                                     mode=mode,
                                     channel_id=channel_id,
                                     wl_bandwidth=wl_bandwidth,
                                     loss=loss,
                                     reflection_target_id=reflection_target_id,
                                     reflectance=reflectance)

    def add_backward_pump(self, wl: float, input_power: float, wl_bandwidth=0.0, loss=None, mode=None, channel_id=None,
                          reflection_target_id=None, reflectance=0.0):
        """Adds a new backward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means seeding by
         spontaneous emission.
        :type wl_bandwidth: float
        :param loss: Background loss of the channel. If None, the fiber's default loss is used.
        :type loss: float
        :param mode: Fiber mode class defining the channel's mode shape.
        :type mode: Subclass of ModeBase (such as LPMode or TophatMode)
        :param channel_id: Identifier for the channel, used for reflection definitions and plotting
        :type channel_id: int or str
        :param reflection_target_id: Identifier for the target channel that this channel reflects to
        :type reflection_target_id: int or str
        :param reflectance: Reflectance at the end of the channel 0<=R<=1
        :type reflectance: float

        """

        self.channels.create_channel(channel_type='pump',
                                     direction=-1,
                                     fiber=self.fiber,
                                     input_power=input_power,
                                     wl=wl,
                                     mode=mode,
                                     channel_id=channel_id,
                                     wl_bandwidth=wl_bandwidth,
                                     loss=loss,
                                     reflection_target_id=reflection_target_id,
                                     reflectance=reflectance)

    def add_ase(self, wl_start, wl_end, n_bins):
        """
        Adds amplified spontaneous emission (ASE) channels.
        Using more channels improves accuracy, but incurs a heavier computational cost to the simulation.

        :param wl_start: The shorted wavelength of the ASE band
        :type wl_start: float
        :param wl_end: The longest wavelength of the ASE band
        :type wl_end: float
        :param n_bins: The number of simulated ASE channels.
        :type n_bins: positive int

        """
        assert wl_end > wl_start, 'End wavelength must be greater than start wavelength.'
        assert isinstance(n_bins, int) and n_bins > 0, 'Number of ASE bins must be a positive integer.'
        ase_wl_bandwidth = (wl_end - wl_start) / n_bins
        ase_wls = np.linspace(wl_start, wl_end, n_bins)
        for wl in ase_wls:
            self.channels.create_channel(channel_type='ase',
                                         direction=1,
                                         fiber=self.fiber,
                                         input_power=SIMULATION_MIN_POWER,
                                         wl=wl,
                                         wl_bandwidth=ase_wl_bandwidth,
                                         mode=None)

            self.channels.create_channel(channel_type='ase',
                                         direction=-1,
                                         fiber=self.fiber,
                                         input_power=SIMULATION_MIN_POWER,
                                         wl=wl,
                                         wl_bandwidth=ase_wl_bandwidth,
                                         mode=None)

    def run(self, tol=1e-3):
        """Runs the simulation, i.e. calculates the steady state of the defined fiber amplifier. ASE or raman
        simulations might require higher tolerance than the default value.
        It is best to decrease the tolerance until the result no longer changes.

        :param tol: Target error tolerance of the solver.
        :type tol: float

        """
        if self.fiber.num_ion_populations > 1:
            raise RuntimeError('Use DynamicSimulation for calculations with multiple transverse ion populations.')
        boundary_condition_residual = self.boundary_conditions(self.channels.get_input_powers(),
                                                               self.channels.get_reflections(),
                                                               self.channels.number_of_forward_channels)
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



