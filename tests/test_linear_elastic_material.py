import unittest

import numpy as np
import xarray as xr

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
    MembraneSection,
    MindlinPlateSection,
    LinearElasticMaterial,
    HoffmanFailureCriterion,
    HoffmanFailureCriterion_SP,
    HoffmanFailureCriterion_M,
    HuberMisesHenckyFailureCriterion,
)
from sigmaepsilon.solid.material import ElasticityTensor
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix


class TestLinearElasticMaterial(SolidMaterialTestCase):
    def test_basic_example_scenario(self):
        material_params = dict(
            E1=2100.0,
            E2=210.0,
            G13=3000.0,
            NU12=0.2,
            NU23=0.02,
            isoplane="23",
        )
        hooke_123 = elastic_stiffness_matrix(
            verify=True,
            **material_params,
        )
        self.assertValidMaterial(hooke_123)
        frame = ReferenceFrame(dim=3)
        stiffness = ElasticityTensor(
            hooke_123, frame=frame, tensorial=False, material_params=material_params
        )

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
        section.utilization(
            strains=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), z=[0], squeeze=False
        )
        section.utilization(
            strains=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), z=[0], squeeze=True
        )
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


class TestLinearElasticMaterialInput(SolidMaterialTestCase):
    def setUp(self):
        self.hooke = elastic_stiffness_matrix(E=210000.0, NU=0.3)
        stiffness = ElasticityTensor(self.hooke, frame=ReferenceFrame(dim=3))
        self.material = LinearElasticMaterial(
            stiffness=stiffness,
            failure_model=HuberMisesHenckyFailureCriterion(yield_strength=355.0),
        )

    def test_array_stiffness(self):
        material = LinearElasticMaterial(stiffness=self.hooke)
        self.assertIs(material.elastic_stiffness_matrix(), self.hooke)

    def test_input_types(self):
        strains = np.random.rand(4, 6) / 1000
        expected = (self.hooke @ strains.T).T
        for _strains in (strains, tuple(strains.T), xr.DataArray(strains)):
            stresses = self.material.calculate_stresses(strains=_strains)
            self.assertEqual(stresses.dims, ("index", "component"))
            self.assertTrue(np.allclose(stresses.values, expected))
        stresses = self.material.calculate_stresses(strains=list(strains[0]))
        self.assertTrue(np.allclose(stresses.values, expected[:1]))
        self.assertIsNone(self.material._input_as_dense_bulk_ndarray(None))

    def test_utilization(self):
        # uniaxial stress at the yield strength
        stresses = np.array([355.0, 0, 0, 0, 0, 0])
        strains = self.material.calculate_strains(stresses=stresses).values
        self.assertEqual(strains.shape, (1, 6))

        utilization = self.material.utilization(strains=np.repeat(strains, 2, axis=0))
        self.assertEqual(utilization.dims, ("index",))
        self.assertTrue(np.allclose(utilization, 1.0))

        utilization = self.material.utilization(strains=strains[0])
        self.assertEqual(utilization.dims, ())
        self.assertTrue(np.allclose(utilization, 1.0))

        utilization = self.material.utilization(stresses=stresses)
        self.assertTrue(np.allclose(utilization, 1.0))


if __name__ == "__main__":
    unittest.main()
