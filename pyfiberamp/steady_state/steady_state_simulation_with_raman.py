from pyfiberamp.helper_funcs import *
from pyfiberamp.simulation_result import SimulationResult
from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.steady_state.models import GilesModelWithRaman


class SteadyStateSimulationWithRaman(SteadyStateSimulation):
    """
    SteadyStateSimulationWithRaman is the main class for running Giles model simulations with Raman scattering.
    Only one ion population is supported. The class defines the fiber, boundary conditions and optical channels used in
    the simulation.
    """
    def __init__(self):
        super().__init__()
        self.raman_is_included = False
        self.model = GilesModelWithRaman

    def add_pulsed_signal(self, wl, power, f_rep, fwhm_duration, wl_bandwidth=0, mode_shape_parameters=None, label=''):
        """Adds a new forward propagating single-frequency pulsed signal to the simulation. A pulsed signal has a higher
        peak power resulting in stronger nonlinear effects, in particular spontaneous and stimulated Raman scattering.
        The pulse shape is assumed to be Gaussian.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input power of the signal at the beginning of the fiber
        :type power: float
        :param f_rep: Repetition frequency of the signal
        :type f_rep: float
        :param fwhm_duration: Full-width at half-maximum duration of the Gaussian pulses
        :type fwhm_duration: float
        :param wl_bandwidth: Wavelength bandwidth of the channel. Finite bandwidth means including ASE.
        :type wl_bandwidth: float
        :param mode_shape_parameters: Defines the mode field shape. Allowed key-value pairs:
         *functional_form* -> ['bessel', 'gaussian', 'tophat']
         *mode_diameter* -> float
         *overlaps* -> list of pre-calculated overlaps between the channel and the ion populations
        :type mode_shape_parameters: dict
        :param label: Optional label for the channel
        :type label: str

        """
        self.channels.add_pulsed_forward_signal(wl, wl_bandwidth, power, f_rep, fwhm_duration,
                                                mode_shape_parameters, label)

    def add_raman(self, input_power=SIMULATION_MIN_POWER, backward_raman_allowed=True, raman_gain=DEFAULT_RAMAN_GAIN):
        """Adds Raman channels to the simulation.

         :param backward_raman_allowed: Determines if only the forward propagating Raman beam is simulated.
         :type backward_raman_allowed: bool, default True
         :param input_power: Input power of the Raman beam(s)
         :type input_power: float, default ~0 W
         :param raman_gain: Raman gain value to be used in the simulation.
         :type raman_gain: float, default 1e-13 m/W

         """
        self.channels.add_raman(input_power, backward_raman_allowed)
        self.raman_is_included = True
        self.model.raman_gain = raman_gain

    def _finalize(self, sol, upper_level_func):
        """Creates the SimulationResult object from the solution object."""
        res = SimulationResult(z=sol.x,
                               powers=sol.y,
                               upper_level_fraction=upper_level_func(sol.y),
                               channels=self.channels,
                               fiber=self.fiber,
                               backward_raman_allowed=self.channels.backward_raman_allowed)
        return res
