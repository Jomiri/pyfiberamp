from .fiber_amplifier_simulation import FiberAmplifierSimulation
from .helper_funcs import *
from .models import GilesModelWithRaman


class FiberAmplifierSimulationWithRaman(FiberAmplifierSimulation):
    def __init__(self, fiber):
        super().__init__(fiber)
        self.raman_is_included = False
        self.model = GilesModelWithRaman

    def add_raman(self, backward_raman_allowed=True, input_power=SIMULATION_MIN_POWER):
        self.channels.add_raman(input_power, backward_raman_allowed)
        self.raman_is_included = True

    def _add_wls_and_slices_to_result(self, res):
        res = super()._add_wls_and_slices_to_result(res)
        res.backward_raman_allowed = self.channels.backward_raman_allowed
        return res
