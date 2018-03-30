from .fiber_base import FiberBase


class PassiveFiber(FiberBase):
    """PassiveFiber describes a step-index single-mode fiber with no dopant ions. It extends the FiberBase class by
    stating that there is no emission or absorption by ions. The only possible gain comes of stimulated Raman
     scattering."""
    def __init__(self, length=0, core_radius=0, background_loss=0, core_na=0):
        super().__init__(length=length,
                         core_radius=core_radius,
                         background_loss=background_loss,
                         core_na=core_na)

    def saturation_parameter(self):
        """Returns the saturation parameter, zeta in Giles mode. For passive fiber it is enough that the parameter is
         non-zero to avoid division by zero."""
        return 1

    def get_channel_gain(self, freq, mode_field_radius):
        """Passive fiber has no gain."""
        return 0

    def get_channel_absorption(self, freq, mode_field_radius):
        """Passive fiber has no absorption by dopant ions."""
        return 0

    def finite_bandwidth_gain(self, center_frequency, frequency_bandwidth, mode_field_radius):
        """Passive fiber has no gain."""
        return 0

    def finite_bandwidth_absorption(self, center_frequency, frequency_bandwidth, mode_field_radius):
        """Passive fiber has no absorption by dopant ions."""
        return 0

