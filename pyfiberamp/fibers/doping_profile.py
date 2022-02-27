import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib import cm
from matplotlib.colors import Normalize

from pyfiberamp.helper_funcs import to_deg


class DopingProfile:
    """
    The DopingProfile class describes the 2D transverse doping parameters of an active fiber in polar coordinates.
    """
    def __init__(self, radii, num_of_angular_sections, ion_number_densities, core_radius):
        """
        Constructor

        :param radii: A list of radii corresponding to concentric radial sections of the doping profile.
        :param num_of_angular_sections: Number of angular sections in the doping profile.
        :param ion_number_densities: List of doping concentrations in the defined sections:
         Number of items: len(radii)*len(num_of_angular_sections)
        :param core_radius: The waveguide core radius of the fiber -> can be different from the doping radii
        """
        self.num_of_total_sections = len(radii) * num_of_angular_sections
        if self.num_of_total_sections != len(ion_number_densities):
            raise RuntimeError(f'{self.num_of_total_sections} doping densities needed,'
                               f' {len(ion_number_densities)} given')
        self.radii = np.hstack((0, np.array(radii)))
        self.angles = np.linspace(0, 2 * np.pi, num_of_angular_sections + 1)
        assert np.all(self.radii[:-1] <= self.radii[1:]), 'The radii must be sorted in ascending order.'
        self.ion_number_densities = np.array(ion_number_densities)
        self.core_radius = core_radius

    def section_radii(self, section_idx):
        """
        Get the radii of a transverse doping profile section.,

        :param section_idx: Index of the section
        :return: Two-element ndarray containing the inner and outer radii of the section
        """
        idx_r = section_idx // (len(self.angles) - 1)
        return self.radii[idx_r:idx_r+2]

    def section_angles(self, section_idx):
        """
        Get the angles defining a transverse doping profile section

        :param section_idx: Index of the section
        :return: Two-element ndarray containing the inner and outer radii of the section.
        """
        idx_angle = section_idx % (len(self.angles) - 1)
        return self.angles[idx_angle:idx_angle+2]

    def section_area(self, section_idx):
        """
        Get the cross sectional area of a doping profile section.

        :param section_idx: Index of the section
        :return: The area
        """
        radii = self.section_radii(section_idx)
        angles = self.section_angles(section_idx)
        return (radii[1]**2 - radii[0]**2) * (angles[1] - angles[0]) / 2

    @property
    def areas(self):
        """
        Get the cross sectional areas of oll the doping profile sections

        :return: A numpy array of the areas ordered by index.
        """
        return np.array([self.section_area(i) for i in range(self.num_of_total_sections)])

    def plot(self):
        """
        Generate a 2D matplotlib plot of the doping profile.
        :return: No return value
        """
        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        xlim = np.array([-self.core_radius, self.core_radius]) * 1.2 * 1e6
        ylim = xlim

        normalizer = Normalize(vmin=np.min(self.ion_number_densities),
                               vmax=np.max(self.ion_number_densities))
        max_density = np.max(self.ion_number_densities)
        cmap = cm.get_cmap('plasma')
        for i, n in enumerate(self.ion_number_densities):
            inner_radius, outer_radius = self.section_radii(i)
            width = outer_radius - inner_radius
            start_angle, stop_angle = self.section_angles(i)
            wedge = Wedge((0, 0), outer_radius * 1e6, to_deg(start_angle), to_deg(stop_angle), width=width * 1e6,
                          facecolor=cmap(normalizer(n)), linestyle='--', linewidth=1, edgecolor=(0, 0, 0, 1))
            ax.add_patch(wedge)

        core_circ = plt.Circle((0, 0),
                               radius=self.core_radius*1e6,
                               facecolor=(0, 0, 0, 0),
                               edgecolor=(0, 0, 0, 1),
                               linewidth=2)
        ax.add_patch(core_circ)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X [um]')
        ax.set_ylabel('Y [um]')
        if normalizer.vmin != normalizer.vmax:
            fig.colorbar(cm.ScalarMappable(norm=normalizer, cmap=cmap), ax=ax)
        plt.show()


if __name__ == '__main__':
    profile = DopingProfile(np.array([1, 3]) * 1e-6,
                            num_of_angular_sections=1,
                            ion_number_densities=np.linspace(0, 2, 2)*1e25,
                            core_radius=6e-6)
    profile.plot()
