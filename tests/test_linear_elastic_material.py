import unittest

import numpy as np

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
    MembraneSection,
    MindlinPlateSection,
    LinearElasticMaterial,
    HoffmanFailureCriterion,
    HoffmanFailureCriterion_SP,
    HoffmanFailureCriterion_M,
)
from sigmaepsilon.solid.material import ElasticityTensor
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix


class TestLinearElasticMaterial(SolidMaterialTestCase):
    def test_basic_example_scenario(self):
        hooke_123 = elastic_stiffness_matrix(
            E1=2100.0,
            E2=210.0,
            G13=3000.0,
            NU12=0.2,
            NU23=0.02,
            isoplane="23",
            verify=True,
        )
        self.assertValidMaterial(hooke_123)
        frame = ReferenceFrame(dim=3)
        stiffness = ElasticityTensor(hooke_123, frame=frame, tensorial=False)

        failure_model = HoffmanFailureCriterion(params=[1.0 for _ in range(9)])

        material = LinearElasticMaterial(
            stiffness=stiffness, failure_model=failure_model
        )

        material.elastic_stiffness_matrix()
        material.calculate_strains(stresses=np.random.rand(6))
        material.calculate_stresses(strains=np.random.rand(6))

        material.utilization(stresses=np.random.rand(6))
        material.utilization(stresses=np.random.rand(2, 6))
        material.utilization(stresses=tuple(np.random.rand(2) for _ in range(6)))

        self.assertIsInstance(material.stiffness, ElasticityTensor)
        self.assertIsInstance(material.failure_model, HoffmanFailureCriterion)

        material.stiffness = material.stiffness
        material.failure_model = material.failure_model

    def test_basic_example_scenario_SP(self):
        hooke_123 = elastic_stiffness_matrix(
            E1=2100.0,
            E2=210.0,
            G13=3000.0,
            NU12=0.2,
            NU23=0.02,
            isoplane="23",
            verify=True,
        )
        self.assertValidMaterial(hooke_123)
        frame = ReferenceFrame(dim=3)
        stiffness = ElasticityTensor(hooke_123, frame=frame, tensorial=False)

        failure_model = HoffmanFailureCriterion_SP(params=[1.0 for _ in range(9)])

        material = LinearElasticMaterial(
            stiffness=stiffness, failure_model=failure_model
        )

        self.assertIsInstance(material.stiffness, ElasticityTensor)
        self.assertIsInstance(material.failure_model, HoffmanFailureCriterion_SP)

        material.stiffness = material.stiffness
        material.failure_model = material.failure_model

        section = MindlinPlateSection(
            layers=[MindlinPlateSection.Layer(material=material, thickness=0.1)]
        )
        section.elastic_stiffness_matrix()

        section.utilization(strains=2 * np.random.rand(10, 5) / 1000) * 100
        section.utilization(strains=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), z=[0], squeeze=False)
        section.utilization(strains=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), z=[0], squeeze=True)
        section.utilization(
            strains=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), z=[-1, 0, 1], squeeze=False
        )
        section.utilization(
            strains=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), z=[-1, 0, 1], squeeze=True
        )
        section.utilization(strains=2 * np.random.rand(5, 5) / 1000) * 100

    def test_basic_example_scenario_M(self):
        hooke_123 = elastic_stiffness_matrix(
            E1=2100.0,
            E2=210.0,
            G13=3000.0,
            NU12=0.2,
            NU23=0.02,
            isoplane="23",
            verify=True,
        )
        self.assertValidMaterial(hooke_123)
        frame = ReferenceFrame(dim=3)
        stiffness = ElasticityTensor(hooke_123, frame=frame, tensorial=False)

        failure_model = HoffmanFailureCriterion_M(params=[1.0 for _ in range(7)])

        material = LinearElasticMaterial(
            stiffness=stiffness, failure_model=failure_model
        )

        self.assertIsInstance(material.stiffness, ElasticityTensor)
        self.assertIsInstance(material.failure_model, HoffmanFailureCriterion_M)

        material.stiffness = material.stiffness
        material.failure_model = material.failure_model

        section = MembraneSection(
            layers=[MembraneSection.Layer(material=material, thickness=0.1)]
        )
        section.elastic_stiffness_matrix()

        section.utilization(strains=2 * np.random.rand(10, 3) / 1000) * 100
        section.utilization(strains=np.array([1.0, 0.0, 0.0]), z=[0], squeeze=False)
        section.utilization(strains=np.array([1.0, 0.0, 0.0]), z=[0], squeeze=True)
        section.utilization(
            strains=np.array([1.0, 0.0, 0.0]), z=[-1, 0, 1], squeeze=False
        )
        section.utilization(
            strains=np.array([1.0, 0.0, 0.0]), z=[-1, 0, 1], squeeze=True
        )
        section.utilization(strains=2 * np.random.rand(5, 3) / 1000) * 100


if __name__ == "__main__":
    unittest.main()
