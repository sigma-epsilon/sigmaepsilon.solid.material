import unittest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.solid.material import BernoulliFrameSection


class TestBernoulliFrameSection(SigmaEpsilonTestCase):
    @classmethod
    def setUpClass(cls):
        cls.section = BernoulliFrameSection("CHS", d=1.0, t=0.1, n=16)
        cls.props = cls.section.calculate_section_properties()

    def test_bernoulli_frame_section(self):
        num_points = len(self.section.mesh["vertices"])
        self.assertEqual(self.props["stress-factors"].shape, (6, 6, num_points))
        self.assertFailsProperly(
            NotImplementedError,
            self.section.calculate_section_properties,
            separate=True,
        )

    def test_elastic_stiffness_matrix(self):
        section = self.section
        E, nu = 210000.0, 0.3
        G = E / (2 * (1 + nu))
        expected = np.diag(
            [E * section.A, G * section.Ix, E * section.Iy, E * section.Iz]
        )
        self.assertTrue(np.allclose(section.elastic_stiffness_matrix(E, nu), expected))

        # the default material of the section has E=1 and nu=0
        expected = np.diag([section.A, section.Ix / 2, section.Iy, section.Iz])
        self.assertTrue(np.allclose(section.elastic_stiffness_matrix(), expected))

    def test_stresses(self):
        section = self.section
        num_points = len(section.mesh["vertices"])

        # axial force results in uniform normal stresses
        stresses = section.calculate_stresses(np.array([1.0, 0, 0, 0, 0, 0]))
        self.assertEqual(stresses.shape, (num_points, 6))
        self.assertTrue(np.allclose(stresses[:, 0], 1 / section.A))
        self.assertTrue(np.allclose(stresses[:, 1:], 0.0))

        # the extreme fibre stress of a bending moment is M * r / I
        stresses = section.calculate_stresses(np.array([0, 0, 0, 0, 0, 1.0]))
        self.assertAlmostEqual(np.abs(stresses[:, 0]).max(), 0.5 / section.Iz)

    def test_stresses_shapes(self):
        section = self.section
        num_points = len(section.mesh["vertices"])
        nE, nP, nRHS = 3, 2, 4
        forces = np.random.rand(nE, nP, 6, nRHS)

        stresses = section.calculate_stresses(forces)
        self.assertEqual(stresses.shape, (nE, nP, num_points, 6, nRHS))

        stresses_3d = section.calculate_stresses(forces[..., 0])
        self.assertEqual(stresses_3d.shape, (nE, nP, num_points, 6))
        self.assertTrue(np.allclose(stresses_3d, stresses[..., 0]))

        stresses_2d = section.calculate_stresses(forces[0, :, :, 0])
        self.assertEqual(stresses_2d.shape, (nP, num_points, 6))
        self.assertTrue(np.allclose(stresses_2d, stresses[0, ..., 0]))

        self.assertFailsProperly(
            RuntimeError, section.calculate_stresses, np.random.rand(2, 2, 2, 6, 1)
        )


if __name__ == "__main__":
    unittest.main()
