
from pyfiberamp.helper_funcs import *
from enum import Enum


class GainShapes(Enum):
    LINEAR = 1
    EXPONENTIAL = 2
    REABSORBED = 3


DEFAULT_GUESS_PARAMS = {
    'signal': {'gain_db': 10, 'gain_shape': GainShapes.EXPONENTIAL},
    'pump':  {'gain_db': -10, 'gain_shape': GainShapes.EXPONENTIAL},
    'ase': {'output_power': 1e-4, 'gain_shape': GainShapes.EXPONENTIAL},
    'raman': {'output_power': 1e-3, 'gain_shape': GainShapes.EXPONENTIAL}
}


class ChannelGuessParameters:
    @classmethod
    def from_gain_value(cls, gain_db, gain_shape):
        params = cls()
        params.set_gain_shape(gain_shape)
        params.set_gain_db(gain_db)
        return params

    @classmethod
    def from_output_power(cls, output_power, gain_shape):
        params = cls()
        params.set_gain_shape(gain_shape)
        params.set_output_power(output_power)
        return params

    def __init__(self):
        self._output_power_func = None
        self._gain_shape = None

    def set_gain_db(self, gain_db):
        self._output_power_func = lambda input_power: input_power * 10**(gain_db / 10)

    def set_output_power(self, output_power):
        self._output_power_func = lambda x: np.full_like(x, output_power)

    def set_gain_shape(self, gain_shape):
        assert(gain_shape in GainShapes)
        self._gain_shape = gain_shape

    def get_output_power(self, input_power):
        return self._output_power_func(input_power)

    def get_gain_shape(self):
        return self._gain_shape


class GuessParameters:
    def __init__(self):
        self.signal = ChannelGuessParameters.from_gain_value(**DEFAULT_GUESS_PARAMS['signal'])
        self.pump = ChannelGuessParameters.from_gain_value(**DEFAULT_GUESS_PARAMS['pump'])
        self.ase = ChannelGuessParameters.from_output_power(**DEFAULT_GUESS_PARAMS['ase'])
        self.raman = ChannelGuessParameters.from_output_power(**DEFAULT_GUESS_PARAMS['raman'])


class InitialGuessBase:
    def __init__(self):
        self.npoints = START_NODES
        self.input_powers = None
        self.slices = None

    def initialize(self, input_powers, slices):
        self.input_powers = input_powers
        self.slices = slices


class InitialGuessFromArray(InitialGuessBase):
    def __init__(self, guess_array, force_node_number=None):
        super().__init__()
        if force_node_number is None:
            self.array = guess_array
        else:
            self.array = resample_array(guess_array, force_node_number)
        self.npoints = self.array.shape[1]

    def guess_shape(self):
        return self.array.shape

    def as_array(self):
        return self.array


class InitialGuessFromParameters(InitialGuessBase):

    def __init__(self):
        super().__init__()
        self.params = GuessParameters()

    def guess_shape(self):
        return len(self.input_powers), self.npoints

    def as_array(self):
        guess = np.zeros(self.guess_shape())
        forward_signal_powers = self.input_powers[self.slices['forward_signal_slice']]
        backward_signal_powers = self.input_powers[self.slices['backward_signal_slice']]
        forward_pump_powers = self.input_powers[self.slices['forward_pump_slice']]
        backward_pump_powers = self.input_powers[self.slices['backward_pump_slice']]
        forward_ase_powers = self.input_powers[self.slices['forward_ase_slice']]
        backward_ase_powers = self.input_powers[self.slices['backward_ase_slice']]
        forward_raman_powers = self.input_powers[self.slices['forward_raman_slice']]
        backward_raman_powers = self.input_powers[self.slices['backward_raman_slice']]
        guess[self.slices['forward_signal_slice']] = self.make_forward_guess(forward_signal_powers,
                                                                                self.params.signal)
        guess[self.slices['backward_signal_slice']] = self.make_backward_guess(backward_signal_powers,
                                                                               self.params.signal)
        guess[self.slices['forward_pump_slice']] = self.make_forward_guess(forward_pump_powers,
                                                                           self.params.pump)
        guess[self.slices['backward_pump_slice']] = self.make_backward_guess(backward_pump_powers,
                                                                             self.params.pump)
        guess[self.slices['forward_ase_slice']] = self.make_forward_guess(forward_ase_powers,
                                                                          self.params.ase)
        guess[self.slices['backward_ase_slice']] = self.make_backward_guess(backward_ase_powers,
                                                                            self.params.ase)
        guess[self.slices['forward_raman_slice']] = self.make_forward_guess(forward_raman_powers,
                                                                            self.params.raman)
        guess[self.slices['backward_raman_slice']] = self.make_backward_guess(backward_raman_powers,
                                                                              self.params.raman)
        return guess

    def make_backward_guess(self, input_power, params):
        return np.fliplr(self.make_forward_guess(input_power, params))

    def make_forward_guess(self, input_power, params):
        output_power = params.get_output_power(input_power)
        shape = params.get_gain_shape()
        if shape == GainShapes.LINEAR:
            guess = self._linear_guess(input_power, output_power)
        elif shape == GainShapes.EXPONENTIAL:
            guess = self._exponential_guess(input_power, output_power)
        else:
            raise RuntimeError('Unrecognized signal gain shape parameter.')
        return guess

    def _linear_guess(self, start, end):
        return linspace_2d(start, end, self.npoints)

    def _exponential_guess(self, start, end):
        start = start[:, np.newaxis]
        end = end[:, np.newaxis]
        gain = np.log(end / start)
        return start * np.exp(gain * np.linspace(0, 1, self.npoints))
