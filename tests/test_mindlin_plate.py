import unittest
from numbers import Number

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import MindlinPlateSection as Section
from sigmaepsilon.solid.material.warnings import SigmaEpsilonMaterialWarning


class TestMindlinPlateSection(SolidMaterialTestCase):
    def test_mindlin_plate_behaviour(self):
        section = Section(
            layers=[
                Section.Layer(
                    material=Section.Material(E=2100000, nu=0.3), thickness=0.1
                )
            ]
        )
        self.assertIsInstance(section.thickness, Number)
        self.assertIsInstance(section.eccentricity, Number)
        self.assertFailsProperly(Exception, setattr, section, "eccentricity", 1.0)
        self.assertFailsProperly(Exception, setattr, section, "eccentricity", "a")

        layer = section.layers[0]
        self.assertIsInstance(layer.thickness, Number)
        self.assertIsInstance(layer.angle, Number)
        layer.material = layer.material
        self.assertFailsProperly(TypeError, setattr, layer, "material", "a")

    def test_mindlin_plate_section_1(self):
        section = Section(
            layers=[
                Section.Layer(
                    material=Section.Material(E=2100000, nu=0.3), thickness=0.1
                )
            ]
        )
        ABDS = section.elastic_stiffness_matrix()
        self.assertEqual(ABDS.shape, (5, 5))
        self.assertValidMaterial(ABDS)

    def test_mindlin_plate_section_warning_1(self):
        section = Section(
            layers=[
                Section.Layer(
                    material=Section.Material(E=2100000, nu=0.3), thickness=0.1
                ),
                Section.Layer(
                    material=Section.Material(
                        E1=2100000,
                        E2=5000,
                        E3=10,
                        nu12=0.3,
                        G13=3000.0,
                        isoplane="12",
                    ),
                    thickness=0.1,
                    angle=30.0,
                ),
            ]
        )

        self.assertWarns(SigmaEpsilonMaterialWarning, section.elastic_stiffness_matrix)
        ABDS = section.elastic_stiffness_matrix()
        self.assertEqual(ABDS.shape, (5, 5))
        self.assertValidMaterial(ABDS)

    def test_mindlin_plate_section_warning_2(self):
        material1 = Section.Material(E=2100000, nu=0.3)
        material2 = Section.Material(E=21000, nu=0.3)

        section = Section(
            layers=[
                Section.Layer(material=material1, thickness=0.1),
                Section.Layer(material=material2, thickness=0.1),
            ]
        )
        
        self.assertWarns(SigmaEpsilonMaterialWarning, section.elastic_stiffness_matrix)
        ABDS = section.elastic_stiffness_matrix()
        self.assertEqual(ABDS.shape, (5, 5))
        self.assertValidMaterial(ABDS)


if __name__ == "__main__":
    unittest.main()
