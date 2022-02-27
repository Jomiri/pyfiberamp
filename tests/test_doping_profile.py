import unittest
import numpy as np

from pyfiberamp.fibers.doping_profile import DopingProfile


class DopingProfileTestCase(unittest.TestCase):
    def test_single_section(self):
        expected_radii = np.array([0, 3e-6])
        expected_angles = np.array([0, 2*np.pi])
        dp = DopingProfile(radii=[3e-6], num_of_angular_sections=1, ion_number_densities=[1], core_radius=3e-6)
        self.assertTrue(np.allclose(dp.radii, expected_radii))
        self.assertTrue(np.allclose(dp.angles, expected_angles))
        self.assertTrue(np.allclose(dp.section_radii(0), expected_radii))
        self.assertTrue(np.allclose(dp.section_angles(0), expected_angles))
        self.assertAlmostEqual(dp.section_area(0), expected_radii[1]**2 * np.pi)
        core_area = dp.core_radius**2 * np.pi
        self.assertAlmostEqual(np.sum(dp.areas), core_area)

    def test_two_rings(self):
        expected_radii = np.array([0, 3e-6, 5e-6])
        expected_angles = np.array([0, 2*np.pi])
        dp = DopingProfile(radii=[3e-6, 5e-6],
                           num_of_angular_sections=1,
                           ion_number_densities=[1, 2],
                           core_radius=5e-6)
        self.assertTrue(np.allclose(dp.radii, expected_radii))
        self.assertTrue(np.allclose(dp.angles, expected_angles))
        self.assertTrue(np.allclose(dp.section_radii(0), expected_radii[0:2]))
        self.assertTrue(np.allclose(dp.section_radii(1), expected_radii[1:]))
        self.assertTrue(np.allclose(dp.section_angles(0), expected_angles))
        self.assertTrue(np.allclose(dp.section_angles(1), expected_angles))
        core_area = dp.core_radius**2 * np.pi
        self.assertAlmostEqual(np.sum(dp.areas), core_area)

    def test_two_angles(self):
        expected_radii = np.array([0, 3e-6])
        expected_angles = np.array([0, np.pi, 2*np.pi])
        dp = DopingProfile(radii=[3e-6],
                           num_of_angular_sections=2,
                           ion_number_densities=[1, 2],
                           core_radius=5e-6)
        self.assertTrue(np.allclose(dp.radii, expected_radii))
        self.assertTrue(np.allclose(dp.angles, expected_angles))
        self.assertTrue(np.allclose(dp.section_radii(0), expected_radii))
        self.assertTrue(np.allclose(dp.section_radii(1), expected_radii))
        self.assertTrue(np.allclose(dp.section_angles(0), expected_angles[0:2]))
        self.assertTrue(np.allclose(dp.section_angles(1), expected_angles[1:]))
        core_area = dp.core_radius**2 * np.pi
        self.assertAlmostEqual(np.sum(dp.areas), core_area)

    def test_two_rings_and_angles(self):
        expected_radii = np.array([0, 3e-6, 5e-6])
        expected_angles = np.array([0, np.pi, 2*np.pi])
        dp = DopingProfile(radii=[3e-6, 5e-6],
                           num_of_angular_sections=2,
                           ion_number_densities=[1, 2, 3, 4],
                           core_radius=5e-6)
        self.assertTrue(np.allclose(dp.radii, expected_radii))
        self.assertTrue(np.allclose(dp.angles, expected_angles))
        self.assertTrue(np.allclose(dp.section_radii(0), expected_radii[0:2]))
        self.assertTrue(np.allclose(dp.section_radii(1), expected_radii[0:2]))
        self.assertTrue(np.allclose(dp.section_radii(2), expected_radii[1:]))
        self.assertTrue(np.allclose(dp.section_radii(3), expected_radii[1:]))
        self.assertTrue(np.allclose(dp.section_angles(0), expected_angles[0:2]))
        self.assertTrue(np.allclose(dp.section_angles(1), expected_angles[1:]))
        self.assertTrue(np.allclose(dp.section_angles(2), expected_angles[0:2]))
        self.assertTrue(np.allclose(dp.section_angles(3), expected_angles[1:]))

        core_area = dp.core_radius**2 * np.pi
        self.assertAlmostEqual(np.sum(dp.areas), core_area)


