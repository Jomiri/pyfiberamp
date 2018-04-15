from .fiber_amplifier_simulation import FiberAmplifierSimulation
from .helper_funcs import *
from .models import GilesModelWithRaman
from .simulation_result import SimulationResult


class FiberAmplifierSimulationWithRaman(FiberAmplifierSimulation):
    """FiberAmplifierSimulationWithRaman is the main class used for running Giles model simulations with Raman scattering.
    The class defines the fiber, boundary conditions and optical channels used in the simulation."""
    def __init__(self):
        """

        :param fiber: The fiber used in the simulation.
        :type fiber: class instance derived from FiberBase

        """
        super().__init__()
        self.raman_is_included = False
        self.model = GilesModelWithRaman

    def add_pulsed_signal(self, wl, power, f_rep, fwhm_duration, mode_field_diameter=0.0):
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
        :param mode_field_diameter: Mode field diameter of the signal.
         If left undefined, will be calculated using the Petermann II equation.
        :type mode_field_diameter: float, optional

        """
        self.channels.add_pulsed_forward_signal(wl, power, f_rep, fwhm_duration, mode_field_diameter)

    def add_raman(self, input_power=SIMULATION_MIN_POWER, backward_raman_allowed=True):
        """Adds Raman channels to the simulation.

         :param backward_raman_allowed: Determines if only the forward propagating Raman beam is simulated.
         :type backward_raman_allowed: bool, default True
         :param input_power: Input power of the Raman beam(s)
         :type input_power: float, default ~0 W

         """
        self.channels.add_raman(input_power, backward_raman_allowed)
        self.raman_is_included = True

    def _finalize(self, sol, upper_level_func):
        """Creates the SimulationResult object from the solution object."""
        res = SimulationResult(solution=sol,
                               upper_level_func=upper_level_func,
                               slices=self.channels.get_slices(),
                               wavelengths=self.channels.get_wavelengths(),
                               is_passive_fiber=self.fiber.is_passive_fiber(),
                               backward_raman_allowed=self.channels.backward_raman_allowed)
        return res
