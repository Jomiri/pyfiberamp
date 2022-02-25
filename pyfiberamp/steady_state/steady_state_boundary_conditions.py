import numpy as np
from copy import deepcopy

from pyfiberamp.util.sliced_array import SlicedArray


class BasicBoundaryConditions:
    """This class implements the most basic possible boundary conditions in the Giles model:
    all input powers should be those given to the model. Backward progapating beams have their inputs at the end."""
    def __init__(self, expected_powers, reflections, n_forward):
        self.expected_powers = expected_powers
        slices = expected_powers.slices

        def boundary_condition_func(powers_start, powers_end):
            powers_start = SlicedArray(powers_start, slices)
            powers_end = SlicedArray(powers_end, slices)
            current = np.hstack((powers_start.forward_signal,
                                 powers_start.forward_pump,
                                 powers_start.forward_ase,
                                 powers_start.forward_raman,
                                 powers_end.backward_signal,
                                 powers_end.backward_pump,
                                 powers_end.backward_ase,
                                 powers_end.backward_raman))
            # reflections
            p_exp = deepcopy(self.expected_powers)
            for source_idx, target_idx, R in reflections:
                if source_idx < n_forward:
                    p_exp[target_idx] += R * powers_end[source_idx]
                else:
                    p_exp[target_idx] += R * powers_start[source_idx]
            return current - p_exp

        self.boundary_condition_func = boundary_condition_func

    def __call__(self, powers_start, powers_end):
        """Returns the residual that tells how much power_start and power_end deviate from the boundary conditions.
        The bvp solver calls this function.

        :param powers_start: Simulated powers at the start of the fiber
        :type powers_start: 1D numpy array, length = number of optical channels
        :param powers_end: Simulated powers at the end of the fiber
        :type powers_end: 1D numpy array, length = number of optical channels

        """
        return self.boundary_condition_func(powers_start, powers_end)

