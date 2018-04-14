from pyfiberamp.helper_funcs import *
from enum import Enum

from pyfiberamp.sliced_array import SlicedArray


class GainShapes(Enum):
    """This Enum defines the possible functional forms used the construct the initial guess."""
    LINEAR = 1
    EXPONENTIAL = 2
    # REABSORBED = 3


DEFAULT_GUESS_PARAMS = {
    'signal': {'gain_db': 10, 'gain_shape': GainShapes.EXPONENTIAL},
    'pump':  {'gain_db': -10, 'gain_shape': GainShapes.EXPONENTIAL},
    'ase': {'output_power': 1e-4, 'gain_shape': GainShapes.EXPONENTIAL},
    'raman': {'output_power': 1e-3, 'gain_shape': GainShapes.EXPONENTIAL}
}


class GuessParameters:
    """GuessParameters defines the guessed gain and functional form of each channel in the simulation.
    See also docs for :class:`.ChannelGuessParameters`"""
    def __init__(self):
        self.signal = ChannelGuessParameters.from_gain_value(**DEFAULT_GUESS_PARAMS['signal'])
        self.pump = ChannelGuessParameters.from_gain_value(**DEFAULT_GUESS_PARAMS['pump'])
        self.ase = ChannelGuessParameters.from_output_power(**DEFAULT_GUESS_PARAMS['ase'])
        self.raman = ChannelGuessParameters.from_output_power(**DEFAULT_GUESS_PARAMS['raman'])


class ChannelGuessParameters:
    """ChannelGuessParameters defines the guessed gain and the functional form of the power evolution for
    each type of channel (signal, pump, ASE, and Raman). The gain can be defined directly or as the output power.
    The gain guess is stored as a function used to calculate the output power."""

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
        """Set new guessed gain value. Overrides default and previously set gain and output power guesses.

        :param gain_db: Guessed total gain in dB
        :type gain_db: float

        """
        self._output_power_func = lambda input_power: input_power * 10**(gain_db / 10)

    def set_output_power(self, output_power):
        """Set new guessed output power value. Overrides default and previously set gain and output power guesses.

        :param output_power: Guessed output power in W
        :type output_power: float

        """
        self._output_power_func = lambda x: np.full_like(x, output_power)

    def set_gain_shape(self, gain_shape):
        """Set new guessed shape of the power evolution. Overrides default or previously set values.

        :param gain_shape: New guess for the functional form of power evolution
        :type gain_shape: Member of GainShapes Enum

        """
        assert gain_shape in GainShapes
        self._gain_shape = gain_shape

    def get_output_power(self, input_power):
        """Getter for the guessed output power.

        :returns: The guessed output power
        :rtype: float

        """
        return self._output_power_func(input_power)

    def get_gain_shape(self):
        """Getter for the guessed shape of the function.

        :returns: The guessed functional form
        :rtype: Member of the GainShapes Enum

        """
        return self._gain_shape


class InitialGuessBase:
    def __init__(self):
        self.npoints = START_NODES
        self.input_powers = None

    def initialize(self, input_powers):
        self.input_powers = input_powers


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
        guess = SlicedArray(np.zeros(self.guess_shape()), self.input_powers.slices)
        guess.forward_signal = self.make_forward_guess(self.input_powers.forward_signal, self.params.signal)
        guess.backward_signal = self.make_backward_guess(self.input_powers.backward_signal, self.params.signal)
        guess.forward_pump = self.make_forward_guess(self.input_powers.forward_pump, self.params.pump)
        guess.backward_pump = self.make_backward_guess(self.input_powers.backward_pump, self.params.pump)
        guess.forward_ase = self.make_forward_guess(self.input_powers.forward_ase, self.params.ase)
        guess.backward_ase = self.make_backward_guess(self.input_powers.backward_ase, self.params.ase)
        guess.forward_raman = self.make_forward_guess(self.input_powers.forward_raman, self.params.raman)
        guess.backward_raman = self.make_backward_guess(self.input_powers.backward_raman, self.params.raman)
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
        return expspace_2d(start, end, self.npoints)

