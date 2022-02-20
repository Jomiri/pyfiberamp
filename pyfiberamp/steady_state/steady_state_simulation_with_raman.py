from functools import partial
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
    def __init__(self, fiber):
        super().__init__(fiber)
        self.raman_is_included = False
        self.model = GilesModelWithRaman

    def add_pulsed_forward_signal(self, wl, input_power, f_rep, fwhm_duration,
                                  wl_bandwidth=0.0, loss=None, mode=None, channel_id=None,
                                  reflection_target_id=None, reflectance=0.0):
        """Adds a new forward propagating single-frequency pulsed signal to the simulation. A pulsed signal has a higher
        peak input_power resulting in stronger nonlinear effects, in particular spontaneous and stimulated Raman scattering.
        The pulse shape is assumed to be Gaussian.

        :param wl: Wavelength of the signal
        :type wl: float
        :param power: Input input_power of the signal at the beginning of the fiber
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
        check_signal_reprate(f_rep)
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
                                     reflectance=reflectance,
                                     peak_power_func=partial(gaussian_peak_power, (f_rep, fwhm_duration)))

    def add_raman(self, input_power=SIMULATION_MIN_POWER, backward_raman_allowed=True, raman_gain=DEFAULT_RAMAN_GAIN,
                  loss=None, mode=None, channel_id=None):
        """Adds Raman channels to the simulation.

         :param backward_raman_allowed: Determines if only the forward propagating Raman beam is simulated.
         :type backward_raman_allowed: bool, default True
         :param input_power: Input input_power of the Raman beam(s)
         :type input_power: float, default ~0 W
         :param raman_gain: Raman gain value to be used in the simulation.
         :type raman_gain: float, default 1e-13 m/W

         """
        assert len(self.channels.forward_signals) == 1 and len(self.channels.backward_signals) == 0, \
            'Raman modeling is supported only with a single forward signal.'

        raman_freq = self.channels.forward_signals[0].v - RAMAN_FREQ_SHIFT
        self.channels.create_channel(channel_type='raman',
                                     direction=1,
                                     fiber=self.fiber,
                                     input_power=input_power,
                                     wl=freq_to_wl(raman_freq),
                                     mode=mode,
                                     channel_id=channel_id,
                                     wl_bandwidth=RAMAN_GAIN_WL_BANDWIDTH,
                                     loss=loss,
                                     num_of_modes=RAMAN_MODES_IN_PM_FIBER)
        self.channels.create_channel(channel_type='raman',
                                     direction=-1,
                                     fiber=self.fiber,
                                     input_power=input_power,
                                     wl=freq_to_wl(raman_freq),
                                     mode=mode,
                                     channel_id=channel_id,
                                     wl_bandwidth=RAMAN_GAIN_WL_BANDWIDTH * backward_raman_allowed,
                                     loss=loss,
                                     num_of_modes=RAMAN_MODES_IN_PM_FIBER)
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
