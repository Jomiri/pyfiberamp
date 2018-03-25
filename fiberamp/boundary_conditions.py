import numpy as np


class BasicBoundaryConditions:
    def __init__(self, channels):
        self.expected_powers = channels.get_input_powers()
        self.slices = channels.get_slices()

        forward_signal_slice = self.slices['forward_signal_slice']
        backward_signal_slice = self.slices['backward_signal_slice']
        forward_pump_slice = self.slices['forward_pump_slice']
        backward_pump_slice = self.slices['backward_pump_slice']
        forward_ase_slice = self.slices['forward_ase_slice']
        backward_ase_slice = self.slices['backward_ase_slice']
        forward_raman_slice = self.slices['forward_raman_slice']
        backward_raman_slice = self.slices['backward_raman_slice']

        def boundary_condition_func(powers_start, powers_end):
            current = np.hstack((powers_start[forward_signal_slice],
                                 powers_end[backward_signal_slice],
                                 powers_start[forward_pump_slice],
                                 powers_end[backward_pump_slice],
                                 powers_start[forward_ase_slice],
                                 powers_end[backward_ase_slice],
                                 powers_start[forward_raman_slice],
                                 powers_end[backward_raman_slice]))
            return current - self.expected_powers

        self.boundary_condition_func = boundary_condition_func

    def __call__(self, powers_start, powers_end):
        return self.boundary_condition_func(powers_start, powers_end)

