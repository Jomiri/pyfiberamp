import numpy as np


class DopingProfile:
    def __init__(self, ion_number_densities, radii=None, areas=None):
        if radii is None and areas is None:
            raise RuntimeError('Either radii or areas must be specified.')
        elif radii is not None and areas is not None:
            raise RuntimeError('Both radii and areas specified. Use only either one.')
        elif radii is None and areas is not None:
            assert len(ion_number_densities) == len(areas)
            self.areas = np.array(areas)
            self.radii = None
        elif radii is not None and areas is None:
            assert len(radii) == len(ion_number_densities)
            self.radii = np.array(radii)
            assert np.all(self.radii[:-1] <= self.radii[1:]), 'The radii must be sorted in ascending order.'
            self.areas = self.calculate_areas(self.radii)

        self.ion_number_densities = np.array(ion_number_densities)

    @staticmethod
    def calculate_areas(radii):
        radii_with_zero = np.hstack((0, radii))
        return np.pi * (radii_with_zero[1:]**2 - radii_with_zero[:-1]**2)



