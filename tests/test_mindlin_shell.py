import unittest
import numpy as np
from numbers import Number

from sigmaepsilon.solid.core.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.core import MindlinShellSection as Section


class TestMindlinShellSection(SolidMaterialTestCase):
    def test_mindlin_shell_behaviour(self):
        section = Section(
            layers=[
                Section.Layer(
                    material=Section.Material(E=2100000, nu=0.3), thickness=0.1
                )
            ]
        )
        self.assertIsInstance(section.thickness, Number)
        self.assertIsInstance(section.eccentricity, Number)
        section.eccentricity = 1.0
        self.assertFailsProperly(TypeError, setattr, section, "eccentricity", "a")

        layer = section.layers[0]
        self.assertIsInstance(layer.thickness, Number)
        self.assertIsInstance(layer.angle, Number)
        layer.material = layer.material
        self.assertFailsProperly(TypeError, setattr, layer, "material", "a")

    def test_mindlin_shell_section_1(self):
        section = Section(
            layers=[
                Section.Layer(
                    material=Section.Material(E=2100000, nu=0.3), thickness=0.1
                )
            ]
        )
        ABDS = section.elastic_stiffness_matrix()
        self.assertEqual(ABDS.shape, (8, 8))
        self.assertLess(np.sum(ABDS[:3, 3:6]), 1e-8)
        self.assertLess(np.sum(ABDS[3:6, :3]), 1e-8)
        self.assertValidMaterial(ABDS)


if __name__ == "__main__":
    unittest.main()
