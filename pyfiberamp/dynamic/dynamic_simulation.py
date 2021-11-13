

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
    def __init__(self, max_time_steps):
        self.fiber = None
        self.channels = Channels()
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
        :param dt: Time step size. The 'auto' option uses realistic time step calculated from the Courant condition \
        based on the speed of light in glass and the spatial step size. Larger (and physically unrealistic) time steps \
        can be used to drastically speed up the convergence of steady state simulations.
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

    def add_forward_signal(self, wl, input_power, wl_bandwidth=0.0, mode_shape_parameters=None, label="", loss=None,
                           reflection_target='', reflectance=0):
        """Adds a new forward-propagating signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input power of the signal at the beginning of the fiber
        :type input_power: float or numpy array
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> one of ['bessel', 'gaussian', 'tophat']  \
         *mode_diameter* -> float \
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel (required to receive reflected power from another channel)
        :type label: str
        :param reflection_target: Label of the channel receiving reflection from this channel
        :type reflection_target: str
        :param reflectance: Reflectance R [0,1] from this channel to the target channel
        """
        self._check_input(wl, input_power, wl_bandwidth, mode_shape_parameters, label, loss, reflection_target, reflectance)
        self.channels.add_forward_signal(wl, wl_bandwidth, input_power, mode_shape_parameters, label, loss,
                                         reflection_target, reflectance)

    def add_backward_signal(self, wl, input_power, wl_bandwidth=0.0, mode_shape_parameters=None, label="", loss=None,
                            reflection_target='', reflectance=0):
        """Adds a new backward-propagating signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input power of the signal at the beginning of the fiber
        :type input_power: float or numpy array
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> one of ['bessel', 'gaussian', 'tophat']  \
         *mode_diameter* -> float \
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel (required to receive reflected power from another channel)
        :type label: str
        :param reflection_target: Label of the channel receiving reflection from this channel
        :type reflection_target: str
        :param reflectance: Reflectance R [0,1] from this channel to the target channel
        """
        self._check_input(wl, input_power, wl_bandwidth, mode_shape_parameters, label, loss, reflection_target, reflectance)
        self.channels.add_backward_signal(wl, wl_bandwidth, input_power, mode_shape_parameters, label, loss,
                                          reflection_target, reflectance)

    def add_forward_pump(self, wl, input_power, wl_bandwidth=0.0, mode_shape_parameters=None, label="", loss=None,
                         reflection_target='', reflectance=0):
        """Adds a new forward-propagating pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input power of the signal at the beginning of the fiber
        :type input_power: float or numpy array
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> one of ['bessel', 'gaussian', 'tophat']  \
         *mode_diameter* -> float \
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel (required to receive reflected power from another channel)
        :type label: str
        :param reflection_target: Label of the channel receiving reflection from this channel
        :type reflection_target: str
        :param reflectance: Reflectance R [0,1] from this channel to the target channel
        """
        self._check_input(wl, input_power, wl_bandwidth, mode_shape_parameters, label, loss, reflection_target, reflectance)
        self.channels.add_forward_pump(wl, wl_bandwidth, input_power, mode_shape_parameters, label, loss,
                                       reflection_target, reflectance)

    def add_backward_pump(self, wl, input_power, wl_bandwidth=0.0, mode_shape_parameters=None, label='', loss=None,
                          reflection_target='', reflectance=0):
        """Adds a new backward-propagating pump to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param input_power: Input power of the signal at the beginning of the fiber
        :type input_power: float or numpy array
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> one of ['bessel', 'gaussian', 'tophat']  \
         *mode_diameter* -> float \
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel (required to receive reflected power from another channel)
        :type label: str
        :param reflection_target: Label of the channel receiving reflection from this channel
        :type reflection_target: str
        :param reflectance: Reflectance R [0,1] from this channel to the target channel
        """
        self._check_input(wl, input_power, wl_bandwidth, mode_shape_parameters, label, loss, reflection_target, reflectance)
        self.channels.add_backward_pump(wl, wl_bandwidth, input_power, mode_shape_parameters, label, loss,
                                        reflection_target, reflectance)

    def add_ase(self, wl_start, wl_end, n_bins):
        """Adds amplified spontaneous emission (ASE) channels.
        Using more channels improves accuracy, but incurs a heavier computational cost to the simulation.

        :param wl_start: The shorted wavelength of the ASE band
        :type wl_start: float
        :param wl_end: The longest wavelength of the ASE band
        :type wl_end: float
        :param n_bins: The number of simulated ASE channels.
        :type n_bins: positive int

        """
        self.channels.add_ase(wl_start, wl_end, n_bins)

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

        self.channels.set_fiber(self.fiber)
        solver = self.backend(self.channels, self.fiber, z_nodes, self.max_time_steps, dt, P, N2,
                              stop_at_steady_state, steady_state_tolerance, convergence_checking_interval)
        res = solver.run()
        return res

    def _check_input(self, wl, input_power, wl_bandwidth, mode_shape_parameters,
                     label, loss, reflection_target, reflection_coeff):
        assert isinstance(wl, (float, int)) and wl > 0, 'Wavelength must be a positive number.'
        self._check_input_power(input_power)
        assert isinstance(wl_bandwidth, float) and wl_bandwidth >= 0, 'Wavelength bandwidth must be a positive float.'
        assert isinstance(label, str), 'Label must be a string.'
        assert isinstance(reflection_target, str), 'Reflection target label must be a string.'
        assert 0 <= reflection_coeff <= 1, 'Reflectance must be between 0 and 1.'
        assert loss is None or loss >= 0, 'Background loss must be >=0'

    def _check_input_power(self, input_power):
        assert (isinstance(input_power, (float, int)) and input_power >= 0) or \
               (isinstance(input_power, np.ndarray) and input_power.shape == (self.max_time_steps,))

