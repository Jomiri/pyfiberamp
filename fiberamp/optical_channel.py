from .helper_funcs import *


class OpticalChannel:
    def __init__(self, v, dv, input_power, direction, mfd, gain, absorption, loss):
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

