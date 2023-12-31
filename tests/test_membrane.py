import unittest
from numbers import Number

import numpy as np

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import MembraneSection as Section
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.solid.material import ElasticityTensor
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix
from sigmaepsilon.solid.material.warnings import SigmaEpsilonMaterialWarning


class TestMembraneSection(SolidMaterialTestCase):
    def test_membrane_behaviour(self):
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

    def test_membrane_section_1(self):
        section = Section(
            layers=[
                Section.Layer(
                    material=Section.Material(E=2100000, nu=0.3), thickness=0.1
                )
            ]
        )
        ABDS = section.elastic_stiffness_matrix()
        self.assertEqual(ABDS.shape, (3, 3))
        self.assertValidMaterial(ABDS)

    def test_membrane_section_warning_1(self):
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
        self.assertEqual(ABDS.shape, (3, 3))
        self.assertValidMaterial(ABDS)

    def test_membrane_section_warning_2(self):
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
        self.assertEqual(ABDS.shape, (3, 3))
        self.assertValidMaterial(ABDS)
        
    def test_membrane_utilization(self):
        yield_strength=355.0
        hooke = elastic_stiffness_matrix(E=210000, NU=0.3)
        frame = ReferenceFrame(dim=3)
        tensor = ElasticityTensor(hooke, frame=frame, tensorial=False, yield_strength=yield_strength)

        section = Section(
            layers=[Section.Layer(material=tensor, thickness=0.1)]
        )
        section.elastic_stiffness_matrix()
        
        section.utilization(strains=2*np.random.rand(10, 3)/1000)*100
        section.utilization(strains=np.array([1.0, 0.0, 0.0]), z=[0], squeeze=False)
        section.utilization(strains=np.array([1.0, 0.0, 0.0]), z=[0], squeeze=True)
        section.utilization(strains=np.array([1.0, 0.0, 0.0]), z=[-1, 0, 1], squeeze=False)
        section.utilization(strains=np.array([1.0, 0.0, 0.0]), z=[-1, 0, 1], squeeze=True)
        section.utilization(strains=2*np.random.rand(5, 3)/1000)*100

if __name__ == "__main__":
    unittest.main()
