import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib import cm
from matplotlib.colors import Normalize


def to_deg(x):
    return x / np.pi * 180


class DopingProfile:
    def __init__(self, radii, num_of_angular_sections, ion_number_densities, core_radius):
        self.num_of_total_sections = len(radii) * num_of_angular_sections
        if self.num_of_total_sections != len(ion_number_densities):
            raise RuntimeError(f'{self.num_of_total_sections} doping densities needed, {len(ion_number_densities)} given')
        self.radii = np.hstack((0, np.array(radii)))
        self.angles = np.linspace(0, 2 * np.pi, num_of_angular_sections + 1)
        assert np.all(self.radii[:-1] <= self.radii[1:]), 'The radii must be sorted in ascending order.'
        self.ion_number_densities = np.array(ion_number_densities)
        self.core_radius = core_radius

    def section_radii(self, section_idx):
        idx_r = section_idx // (len(self.angles) - 1)
        return self.radii[idx_r:idx_r+2]

    def section_angles(self, section_idx):
        idx_angle = section_idx % (len(self.angles) - 1)
        return self.angles[idx_angle:idx_angle+2]

    def section_area(self, section_idx):
        radii = self.section_radii(section_idx)
        angles = self.section_angles(section_idx)
        return (radii[1]**2 - radii[0]**2) * (angles[1] - angles[0]) / 2

    @property
    def areas(self):
        return np.array([self.section_area(i) for i in range(self.num_of_total_sections)])

    def plot(self):
        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        xlim = np.array([-self.core_radius, self.core_radius]) * 1.2 * 1e6
        ylim = xlim

        normalizer = Normalize(vmin=np.min(self.ion_number_densities), vmax=np.max(self.ion_number_densities))
        max_density = np.max(self.ion_number_densities)
        cmap = cm.get_cmap('plasma')
        for i, n in enumerate(self.ion_number_densities):
            inner_radius, outer_radius = self.section_radii(i)
            width = outer_radius - inner_radius
            start_angle, stop_angle = self.section_angles(i)
            wedge = Wedge((0, 0), outer_radius*1e6, to_deg(start_angle), to_deg(stop_angle), width=width*1e6,
                          facecolor=cmap(normalizer(n)), linestyle='--', linewidth=1, edgecolor=(0, 0, 0, 1))
            ax.add_patch(wedge)
            print(inner_radius, outer_radius, start_angle, stop_angle)

        core_circ = plt.Circle((0, 0), radius=self.core_radius*1e6, facecolor=(0, 0, 0, 0), edgecolor=(0, 0, 0, 1), linewidth=2)
        ax.add_patch(core_circ)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X [um]')
        ax.set_ylabel('Y [um]')
        fig.colorbar(cm.ScalarMappable(norm=normalizer, cmap=cmap), ax=ax)
        plt.show()


if __name__ == '__main__':
    profile = DopingProfile(np.array([1, 3]) * 1e-6,
                            num_of_angular_sections=1,
                            ion_number_densities=np.linspace(0, 2, 2)*1e25,
                            core_radius=6e-6)
    profile.plot()




"""
    @staticmethod
    def calculate_areas(radii):
        radii_with_zero = np.hstack((0, radii))
        return np.pi * (radii_with_zero[1:]**2 - radii_with_zero[:-1]**2)
"""
