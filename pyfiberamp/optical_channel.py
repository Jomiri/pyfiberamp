import inspect

from .helper_funcs import *


class OpticalChannel:
    """
    The OpticalChannel class represents in single one-way power signal propagating in the fiber. It contains both
    user-supplied and pre-computed values required by the simulation model.

    """
    @classmethod
    def from_mode(cls, channel_id, channel_type: str, direction: int,
                  fiber, input_power: float, wl: float, mode,
                  num_of_modes=NUMBER_OF_ASE_POLARIZATION_MODES, wl_bandwidth=0,
                  loss=None,
                  reflection_target_id=None, reflection_coeff=0,
                  peak_power_func=lambda x: x):
        """
        Create an OpticalChannel based on a defined fiber mode. This is the highest-level method for channel creation. The mode shape is used to compute its overlap with the core and then gain and absorption in the fiber.

        :param channel_id: Identifier for the channel, used for defining reflections and accessing the channel later
        :type channel_id: str or int
        :param channel_type: The kind of channel ('signal', 'pump', 'ase' or 'raman') is used for generating an initial guess for the steady state algorithm. It is also used for grouping channels for plotting.
        :param direction: Either +1 = forward or -1 = backward
        :param fiber: The fiber used in the simulation (subclass of FiberBase)
        :param input_power: Channel power at the input.
        :param wl: Wavelength of the channel
        :param mode: Fiber mode representing the channel (subclass of ModeBase)
        :param num_of_modes: Only relevant for ASE generation, normally = 2 for two polarization modes.
        :param wl_bandwidth: Wavelength bandwidth of the channel. Only relevant for ASE generation. Large bandwidths should be divided into multiple channels.
        :param loss: Background loss of the channel in 1/m.
        :param reflection_target_id: If this channel reflects to a another channel, give that channel's id here.
        :param reflection_coeff: Reflectance to that other channel.
        :param peak_power_func: Function mapping average power to peak power for pulsed signals. Only relevant for Raman.
        :return: OpticalChannel

        """
        overlaps = np.array([mode.core_section_overlap(r_lim=fiber.doping_profile.section_radii(i),
                                                       phi_lim=fiber.doping_profile.section_angles(i))
                             for i in range(fiber.doping_profile.num_of_total_sections)])
        return OpticalChannel.from_overlaps(channel_id, channel_type, direction,
                                            fiber, input_power, wl, overlaps, num_of_modes, wl_bandwidth, loss,
                                            reflection_target_id, reflection_coeff, peak_power_func, mode=mode)

    @classmethod
    def from_overlaps(cls, channel_id, channel_type, direction,
                      fiber, input_power, wl, overlaps,
                      num_of_modes=NUMBER_OF_ASE_POLARIZATION_MODES, wl_bandwidth=0,
                      loss=None,
                      reflection_target_id=None, reflection_coeff=0,
                      peak_power_func=lambda x: x,
                      mode=None):
        """
        Create an OpticalChannel based on pre-computed overlap(s) with the ion populations in the fiber core. This is the second highest-level method for channel creation. The overlap(s) are used to calculate gain and absorption in the fiber.

        :param channel_id: Identifier for the channel, used for defining reflections and accessing the channel later
        :type channel_id: str or int
        :param channel_type: The kind of channel ('signal', 'pump', 'ase' or 'raman') is used for generating an initial guess for the steady state algorithm. It is also used for grouping channels for plotting.
        :param direction: Either +1 = forward or -1 = backward
        :param fiber: The fiber used in the simulation (subclass of FiberBase)
        :param input_power: Channel power at the input.
        :param wl: Wavelength of the channel
        :param overlaps: Numpy array of the channel's overlaps with each ion population in the fiber's core. Refer to DopingProfile for the definition of these ion populations.
        :param num_of_modes: Only relevant for ASE generation, normally = 2 for two polarization modes.
        :param wl_bandwidth: Wavelength bandwidth of the channel. Only relevant for ASE generation. Large bandwidths should be divided into multiple channels.
        :param loss: Background loss of the channel in 1/m.
        :param reflection_target_id: If this channel reflects to a another channel, give that channel's id here.
        :param reflection_coeff: Reflectance to that other channel.
        :param peak_power_func: Function mapping average power to peak power for pulsed signals. Only relevant for Raman.
        :return: OpticalChannel

        """
        center_frequency = wl_to_freq(wl)
        frequency_bandwidth = wl_bw_to_freq_bw(wl_bandwidth, wl)
        gain = np.array((overlaps *
                fiber.get_channel_emission_cross_section(center_frequency, frequency_bandwidth)
                * fiber.doping_profile.ion_number_densities))
        absorption = np.array((overlaps
                      * fiber.get_channel_absorption_cross_section(center_frequency, frequency_bandwidth)
                      * fiber.doping_profile.ion_number_densities))
        if loss is None:
            loss = np.array(fiber.background_loss)
        return OpticalChannel.from_gain_and_absorption(channel_id, channel_type, direction,
                                                       input_power, wl, gain, absorption, loss,
                                                       num_of_modes, wl_bandwidth,
                                                       reflection_target_id, reflection_coeff,
                                                       peak_power_func, mode=mode, overlaps=overlaps)

    @classmethod
    def from_gain_and_absorption(cls, channel_id, channel_type, direction,
                                 input_power, wl, gain, absorption, loss,
                                 num_of_modes=NUMBER_OF_ASE_POLARIZATION_MODES, wl_bandwidth=0,
                                 reflection_target_id=None, reflection_coeff=0,
                                 peak_power_func=lambda x: x,
                                 mode=None,
                                 overlaps=None):
        """
        Create an OpticalChannel based on pre-computed gain and absorption values.
        This is the third highest-level method for channel creation.

        :param channel_id: Identifier for the channel, used for defining reflections and accessing the channel later
        :type channel_id: str or int
        :param channel_type: The kind of channel ('signal', 'pump', 'ase' or 'raman') is used for generating an initial guess for the steady state algorithm. It is also used for grouping channels for plotting.
        :param direction: Either +1 = forward or -1 = backward
        :param fiber: The fiber used in the simulation (subclass of FiberBase)
        :param input_power: Channel power at the input.
        :param wl: Wavelength of the channel
        :param gain: Numpy array of the channel's gain from each ion population in the fiber's core.
        :param absorption: Numpy array of the channel's absorption from each ion population in the fiber's core.
        :param num_of_modes: Only relevant for ASE generation, normally = 2 for two polarization modes.
        :param wl_bandwidth: Wavelength bandwidth of the channel. Only relevant for ASE generation. Large bandwidths should be divided into multiple channels.
        :param loss: Background loss of the channel in 1/m.
        :param reflection_target_id: If this channel reflects to a another channel, give that channel's id here.
        :param reflection_coeff: Reflectance to that other channel.
        :param peak_power_func: Function mapping average power to peak power for pulsed signals. Only relevant for Raman.
        :return: OpticalChannel

        """
        center_frequency = wl_to_freq(wl)
        frequency_bandwidth = wl_bw_to_freq_bw(wl_bandwidth, wl)
        center_frequency = np.full_like(gain, center_frequency)
        frequency_bandwidth = np.full_like(gain, frequency_bandwidth)
        loss_final = np.full_like(gain, loss)
        return OpticalChannel(channel_id, channel_type, direction,
                              input_power, center_frequency,
                              num_of_modes=num_of_modes,
                              frequency_bandwidth=frequency_bandwidth,
                              mode=mode,
                              overlaps=overlaps,
                              gain=gain, absorption=absorption, loss=loss_final,
                              reflection_target_id=reflection_target_id, reflectance=reflection_coeff,
                              peak_power_func=peak_power_func)

    def __init__(self, channel_id, channel_type, direction,
                 input_power, center_freq,
                 num_of_modes=NUMBER_OF_ASE_POLARIZATION_MODES, frequency_bandwidth=np.array([0.0]),
                 mode=None,
                 overlaps=None,
                 gain=np.array([0]), absorption=np.array([0]), loss=np.array([0]),
                 reflection_target_id=None, reflectance=0,
                 peak_power_func=lambda x: x):
        input_power = np.array(input_power)
        self._check_input(center_freq, input_power, frequency_bandwidth, loss, reflection_target_id, reflectance)
        self.channel_id = channel_id
        self.channel_type = channel_type
        self.direction = direction

        min_clamp(input_power, SIMULATION_MIN_POWER)
        self.input_power = input_power
        self.v = center_freq

        # Parameters for ASE calculation
        self.number_of_modes = num_of_modes
        self.dv = frequency_bandwidth

        self.gain = gain
        self.absorption = absorption
        self.loss = loss

        self.mode = mode
        self.overlaps = overlaps

        # For end reflections
        self.reflection_target_id = reflection_target_id
        self.end_reflection_coeff = reflectance

        # For Raman calculation based on peak input_power
        self.peak_power_func = peak_power_func

    def __str__(self):
        return inspect.cleandoc(f"""
        --- Optical channel ---
        ID: {self.channel_id} 
        Type: {self.channel_type} 
        Direction: {'forward' if self.direction==1 else 'backward'}
        Input power: {self.input_power}
        Center_wavelength: {self.wavelength}
        Gain: {self.gain}
        Absorption: {self.absorption}
        Loss: {self.loss}
        Mode: {str(self.mode)}
        Reflection target ID: {self.reflection_target_id}
        Reflectance: {self.end_reflection_coeff}
         """)

    @property
    def wavelength(self):
        return freq_to_wl(self.v[0])

    @property
    def wavelength_bandwidth(self):
        return freq_bw_to_wl_bw(self.dv, self.wavelength)

    @staticmethod
    def _check_input(freq, input_power, freq_bandwidth, loss, reflection_target, reflection_coeff):
        assert (isinstance(freq, (float, int)) and freq > 0) or (isinstance(freq, np.ndarray) and freq[0] > 0), \
            'Wavelength must be a positive number.'
        assert np.all(input_power >= 0)
        assert (isinstance(freq_bandwidth, (float, int))
                and freq_bandwidth) >= 0, 'Wavelength bandwidth must be a positive float.'
        assert reflection_target is None or isinstance(reflection_target, (str, int)), \
            'Reflection target label must be a string or an int.'
        assert 0 <= reflection_coeff <= 1, 'Reflectance must be between 0 and 1.'
        assert loss is None or (isinstance(loss, (int, float)) and loss >= 0) or (isinstance(loss, np.ndarray)
                                                                                  and loss[0] >= 0), \
            'Background loss must be >=0'

