import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from pyfiberamp.plotting.plotter import plot_dynamic_outputs

from pyfiberamp.simulation_result import SimulationResult


class DynamicSimulationResult(SimulationResult):
    def __init__(self, z, t, powers, upper_level_fraction, output_powers, channels, fiber):
        super().__init__(z, powers, upper_level_fraction, channels, fiber, backward_raman_allowed=False)
        self.output_powers = output_powers
        self.t = t

    def plot_outputs(self, channel_ids=None, plot_density=1):
        plot_dynamic_outputs(self, channel_ids, plot_density)

    def plot_transverse_inversion(self, z_idx):
        dummy_doping_profile = deepcopy(self.fiber.doping_profile)
        dummy_doping_profile.ion_number_densities = self.upper_level_fraction[:, z_idx]
        dummy_doping_profile.plot()


