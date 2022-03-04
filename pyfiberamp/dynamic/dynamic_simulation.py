

from pyfiberamp.channels import Channels
from pyfiberamp.dynamic.dynamic_solver_python import DynamicSolverPython
from pyfiberamp.helper_funcs import *
import logging


class DynamicSimulation:
    """
     DynamicSimulation is the interface class used for running fiber amplifier simulations with arbitrarily varying input
     powers. It also supports reflective boundary conditions and thus modeling of simple CW, gain-switched or Q-switched
     fiber lasers. With constant input powers, the result converges to the steady state simulation result. Setting
     multiple ion populations is also supported. The class defines the fiber, boundary conditions and optical channels
     used in the simulation.
     """
    def __init__(self, max_time_steps: int, fiber):
        """
        Constructor. The fiber must supplied already at this point and should be changed later.
        :param max_time_steps: The maximum number of time steps the simulation is initialized to run.
        :type max_time_steps: int
        :param fiber: The fiber to be simulated.
        :type fiber: Subclass of FiberBase

        """
        self.fiber = fiber
        self.channels = Channels(fiber)
        self.max_time_steps = int(max_time_steps)
        self.backends = self._get_available_backends()
        self._use_backend(self._fastest_backend())  # use fastest available backend by default

    @staticmethod
    def _get_available_backends():
        """ Returns available dynamic simulation backends ordered from the fastest to the slowest. """
        backends = []
        try:
            from pyfiberamp.dynamic.dynamic_solver_cpp import DynamicSolverCpp
            backends.append('cpp')
        except ModuleNotFoundError:
            pass
        try:
            from pyfiberamp.dynamic.dynamic_solver_pythran import DynamicSolverPythran
            backends.append('pythran')
        except ImportError:
            pass
        try:
            from pyfiberamp.dynamic.dynamic_solver_numba import DynamicSolverNumba
            backends.append('numba')
        except ImportError:
            pass
        backends.append('python')
        return backends

    def _fastest_backend(self):
        return self.backends[0]

    def use_python_backend(self):
        """
        Sets the simulation to use the slow Python finite difference solver. Using one of the faster solvers instead
        is highly recommended.
        """
        self._use_backend('python')

    def use_cpp_backend(self):
        """
        Sets the simulation to use the C++ backend if available.
        """
        self._use_backend('cpp')

    def use_pythran_backend(self):
        """
        Sets the simulation to use the pythran backend if available.
        """
        self._use_backend('pythran')

    def use_numba_backend(self):
        """
        Sets the simulation to use the numba backend if available.
        """
        self._use_backend('numba')

    def _use_backend(self, backend_name):
        if backend_name not in self.backends:
            default_backend = self._fastest_backend()
            logging.warning('Backend {} is not available!\nDetected backends: [{}]\nDefaulting to: {}'
                            .format(backend_name, ', '.join(self.backends), default_backend))
            self._use_backend(default_backend)
        else:
            if backend_name == 'cpp':
                from pyfiberamp.dynamic.dynamic_solver_cpp import DynamicSolverCpp
                self.backend = DynamicSolverCpp
            elif backend_name == 'pythran':
                from pyfiberamp.dynamic.dynamic_solver_pythran import DynamicSolverPythran
                self.backend = DynamicSolverPythran
            elif backend_name == 'numba':
                from pyfiberamp.dynamic.dynamic_solver_numba import DynamicSolverNumba
                self.backend = DynamicSolverNumba
            elif backend_name == 'python':
                self.backend = DynamicSolverPython

    def get_time_coordinates(self, fiber, z_nodes, dt='auto'):
        """
        Returns the time coordinates used in the simulation. Useful for setting time-varying input powers.

        :param fiber: The fiber used in the simulation
        :type fiber: Subclass of FiberBase
        :param z_nodes: Number of spatial nodes used in the simulation.
        :type z_nodes: int
        :param dt: Time step size. The 'auto' option uses realistic time step calculated from the Courant condition based on the speed of light in glass and the spatial step size. Larger (and physically unrealistic) time steps can be used to drastically speed up the convergence of steady state simulations.
        :type dt: float
        :returns: Time coordinate array
        :rtype: numpy float array

        """
        dz = fiber.length / (z_nodes - 1)
        if dt == 'auto':
            cn = c / DEFAULT_GROUP_INDEX
            dt = dz / cn
        else:
            assert dt < self.fiber.spectroscopy.upper_state_lifetime, 'Even for steady state simulation, the time step'\
                                                                      'should be below the upper state life time.'
        return np.linspace(0, self.max_time_steps, self.max_time_steps, endpoint=False) * dt

    def add_forward_signal(self, wl: float, input_power, wl_bandwidth=0.0, loss=None, mode=None, channel_id=None,
                           reflection_target_id=None, reflectance=0.0):
        """Adds a new forward propagating single-frequency CW signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float or np.ndarray
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

    def add_backward_signal(self, wl: float, input_power, wl_bandwidth=0.0, loss=None,
                            mode=None, channel_id=None,
                            reflection_target_id=None, reflectance=0.0):
        """Adds a new forward propagating single-frequency CW signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float or np.ndarray
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means seeding by spontaneous emission.
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

    def add_forward_pump(self, wl:float, input_power, wl_bandwidth=0.0, loss=None, mode=None, channel_id=None,
                         reflection_target_id=None, reflectance=0.0):
        """Adds a new forward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float or np.ndarray
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

    def add_backward_pump(self, wl: float, input_power, wl_bandwidth=0.0, loss=None, mode=None, channel_id=None,
                          reflection_target_id=None, reflectance=0.0):
        """Adds a new backward propagating single-frequency pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input input_power of the signal at the beginning of the fiber
        :type input_power: float or np.ndarray
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
        self._check_input_power(input_power)
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

    def run(self, z_nodes, dt='auto', P=None, N2=None, stop_at_steady_state=False,
            steady_state_tolerance=1e-4, convergence_checking_interval=10000):
        """
        Runs the simulation.

        :param z_nodes: Number of spatial nodes used in the simulation.
        :type z_nodes: int
        :param dt: Time step size. The 'auto' option uses realistic time step calculated from the Courant condition \
        based on the speed of light in glass and the spatial step size. Larger (and physically unrealistic) time steps \
        can be used to drastically speed up the convergence of steady state simulations.
        :type dt: float or str
        :param P: Pre-existing powers in the fiber, useful when chaining multiple simulations.
        :type P: numpy float array
        :param N2: Pre-existing upper state excitation in the fiber, useful when chaining multiple simulations.
        :type N2: numpy float array
        :param stop_at_steady_state: If this flag parameter is set to True, the simulation stops when the excitation \
        reaches a steady state (does not work if the excitation fluctuates at a specific frequency).
        :type stop_at_steady_state: bool
        :param steady_state_tolerance: Sets the relative change in excitation that is used to detect the steady state.
        :type steady_state_tolerance: float
        :param convergence_checking_interval: If aiming for steady state, the simulation checks convergence always after \
        this number of iterations and prints the average excitation. In truly dynamic simulations, only prints the \
        excitation.
        :type convergence_checking_interval: positive int

        """
        solver = self.backend(self.channels, self.fiber, z_nodes, self.max_time_steps, dt, P, N2,
                              stop_at_steady_state, steady_state_tolerance, convergence_checking_interval)
        res = solver.run()
        return res

    def _check_input_power(self, input_power):
        assert (isinstance(input_power, (float, int)) and input_power >= 0) or \
               (isinstance(input_power, np.ndarray) and input_power.shape == (self.max_time_steps,))

