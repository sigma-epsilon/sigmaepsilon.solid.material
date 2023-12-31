import unittest

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import KirchhoffShellSection as Section
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.solid.material import ElasticityTensor
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix


class TestKirchhoffShellSection(SolidMaterialTestCase):
    
    def test_example_1(self):
        E = 2890.0
        nu = 0.2
        yield_strength = 2.0
        thickness = 25.0

        hooke = elastic_stiffness_matrix(E=E, NU=nu)
        frame = ReferenceFrame(dim=3)
        tensor = ElasticityTensor(
            hooke, frame=frame, tensorial=False, yield_strength=yield_strength
        )

        section = Section(
            layers=[
                Section.Layer(material=tensor, thickness=thickness / 3),
                Section.Layer(material=tensor, thickness=thickness / 3),
                Section.Layer(material=tensor, thickness=thickness / 3),
            ]
        )
        
        ABDS = section.elastic_stiffness_matrix()
        self.assertEqual(ABDS.shape, (6, 6))
        self.assertValidMaterial(ABDS)


if __name__ == "__main__":
    unittest.main()