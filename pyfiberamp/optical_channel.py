from .helper_funcs import *


class OpticalChannel:
    def __init__(self, v, dv, input_power, direction, mfd, gain, absorption, loss, label,
                 reflection_target_label="", reflection_coeff=0):
        self.v = v
        self.dv = dv
        self.input_power = input_power
        self.direction = direction
        self.mfd = mfd
        self.gain = gain
        self.absorption = absorption
        self.loss = loss
        self.peak_power_func = lambda x: x
        self.number_of_modes = NUMBER_OF_MODES_IN_SINGLE_MODE_FIBER
        self.label = label
        self.reflection_target = reflection_target_label
        self.reflection_coeff = reflection_coeff



