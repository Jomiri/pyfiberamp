

from pyfiberamp.channels import Channels
from pyfiberamp.dynamic.dynamic_solver_cpp import DynamicSolverCpp
from pyfiberamp.dynamic.dynamic_solver_python import DynamicSolverPython
from pyfiberamp.helper_funcs import *
import warnings


class DynamicSimulation:
    """FiberAmplifierSimulation is the main class used for running Giles model simulations without Raman scattering.
    The class defines the fiber, boundary conditions and optical channels used in the simulation."""
    def __init__(self, max_time_steps, dt='auto'):
        self.fiber = None
        self.channels = Channels()
        self.max_time_steps = max_time_steps
        self.backend = None
        self.use_cpp_backend()
        self.solver = None

    def use_python_backend(self):
        self.backend = DynamicSolverPython

    def use_cpp_backend(self):
        try:
            self.backend = DynamicSolverCpp
        except FileNotFoundError:
            warnings.warn('C++ backend could not be loaded! Defaulting to Python backend.', RuntimeWarning)
            self.use_python_backend()

    def get_time_coordinates(self, fiber, z_nodes, dt='auto'):
        dz = fiber.length / (z_nodes - 1)
        if dt == 'auto':
            cn = c / DEFAULT_GROUP_INDEX
            dt = dz / cn
        else:
            assert dt < self.fiber.spectroscopy.upper_state_lifetime, 'Even for steady state simulation, the time step'\
                                                                      'should be below the upper state life time.'
        return np.linspace(0, self.max_time_steps, self.max_time_steps) * dt

    def add_forward_signal(self, wl, input_power, wl_bandwidth=0.0, mode_field_diameter=0.0, label="",
                                    reflection_target='', reflection_coeff=0):
        """Adds a new forward propagating signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param mode_field_diameter: Mode field diameter of the signal.
         If left undefined, will be calculated using the Petermann II equation.
        :type mode_field_diameter: float, optional

        """
        self._check_input(wl, input_power, wl_bandwidth, mode_field_diameter, label, reflection_target, reflection_coeff)
        self.channels.add_forward_signal(wl, wl_bandwidth, input_power, mode_field_diameter, label, reflection_target,
                                         reflection_coeff)

    def add_backward_signal(self, wl, input_power, wl_bandwidth=0.0, mode_field_diameter=0.0, label="",
                           reflection_target='', reflection_coeff=0):
        self._check_input(wl, input_power, wl_bandwidth, mode_field_diameter, label, reflection_target, reflection_coeff)
        self.channels.add_backward_signal(wl, wl_bandwidth, input_power, mode_field_diameter, label, reflection_target,
                                         reflection_coeff)

    def add_forward_pump(self, wl, input_power, wl_bandwidth=0.0, mode_field_diameter=0.0, label="",
                           reflection_target='', reflection_coeff=0):
        """Adds a new forward propagating signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param mode_field_diameter: Mode field diameter of the signal.
         If left undefined, will be calculated using the Petermann II equation.
        :type mode_field_diameter: float, optional

        """
        self._check_input(wl, input_power, wl_bandwidth, mode_field_diameter, label, reflection_target, reflection_coeff)
        self.channels.add_forward_pump(wl, wl_bandwidth, input_power, mode_field_diameter, label, reflection_target,
                                         reflection_coeff)

    def add_backward_pump(self, wl, input_power, wl_bandwidth=0.0, mode_field_diameter=0.0, label='',
                         reflection_target='', reflection_coeff=0):
        """Adds a new forward propagating signal to the simulation.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param mode_field_diameter: Mode field diameter of the signal.
         If left undefined, will be calculated using the Petermann II equation.
        :type mode_field_diameter: float, optional

        """
        self._check_input(wl, input_power, wl_bandwidth, mode_field_diameter, label, reflection_target, reflection_coeff)
        self.channels.add_backward_pump(wl, wl_bandwidth, input_power, mode_field_diameter, label, reflection_target,
                                       reflection_coeff)

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

    def run(self, z_nodes, dt='auto', P=None, N2=None, stop_at_steady_state=False,
            steady_state_tolerance=1e-4):
        """Runs the simulation, i.e. calculates the steady state of the defined fiber amplifier. ASE or raman
        simulations might require higher tolerance than the default value.
        It is best to decrease the tolerance until the result no longer changes.

        :param tol: Target error tolerance of the solver.
        :type tol: float

        """
        self.channels.set_fiber(self.fiber)
        solver = self.backend(self.channels, self.fiber, z_nodes, self.max_time_steps, dt, P, N2,
                              stop_at_steady_state, steady_state_tolerance)
        res = solver.run()
        return res

    def _check_input(self, wl, input_power, wl_bandwidth, mode_field_diameter, label, reflection_target, reflection_coeff):
        assert isinstance(wl, float) and wl > 0
        self._check_input_power(input_power)
        assert isinstance(wl_bandwidth, float) and wl_bandwidth >= 0
        assert mode_field_diameter >= 0
        assert isinstance(label, str)
        assert isinstance(reflection_target, str)
        assert 0 <= reflection_coeff <= 1

    def _check_input_power(self, input_power):
        assert (isinstance(input_power, (float, int)) and input_power >= 0) or \
               (isinstance(input_power, np.ndarray) and input_power.shape==(self.max_time_steps,))

