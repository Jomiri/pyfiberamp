from .fiber_amplifier_simulation import FiberAmplifierSimulation
from .helper_funcs import *
from .models import GilesModelWithRaman


class FiberAmplifierSimulationWithRaman(FiberAmplifierSimulation):
    """FiberAmplifierSimulation is the main class used for running Giles model simulations with Raman. The class
    defines the fiber, boundary conditions and optical channels used in the simulation."""
    def __init__(self, fiber):
        """
        Parameters
        ----------
        fiber : class instance derived from FiberBase
            The fiber used in the simulation.
        """
        super().__init__(fiber)
        self.raman_is_included = False
        self.model = GilesModelWithRaman

    def add_pulsed_signal(self, wl, power, f_rep, fwhm_duration, mode_field_diameter=0.0):
        """Adds a new forward propagating single-frequency pulsed signal to the simulation. A pulsed signal has a higher
        peak power resulting in stronger nonlinear effects, in particular spontaneous and stimulated Raman scattering.
        The pulse shape is assumed to be Gaussian.

        Parameters
        ----------
        wl : float
            Wavelength of the signal
        power : float
            Input power of the signal at the beginning of the fiber
        f_rep : float
            Repetition frequency of the signal
        fwhm_duration : float
            Full-width at half-maximum duration of the Gaussian pulses
        mode_field_diameter : float (optional)
            Mode field diameter of the signal. If left undefined, will be calculated using the Petermann II equation.
        """
        self.channels.add_pulsed_forward_signal(wl, power, f_rep, fwhm_duration, mode_field_diameter)

    def add_raman(self, backward_raman_allowed=True, input_power=SIMULATION_MIN_POWER):
        """Adds Raman channels to the simulation. """
        self.channels.add_raman(input_power, backward_raman_allowed)
        self.raman_is_included = True

    def _add_wls_and_slices_to_result(self, res):
        res = super()._add_wls_and_slices_to_result(res)
        res.backward_raman_allowed = self.channels.backward_raman_allowed
        return res
