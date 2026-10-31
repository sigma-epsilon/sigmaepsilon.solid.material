import unittest

import numpy as np

from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.math.linalg.exceptions import TensorShapeMismatchError

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
    CauchyStressTensor,
    SmallStrainTensor,
    ElasticityTensor,
)
from sigmaepsilon.solid.material.tensor2x3 import Tensor2x3
from sigmaepsilon.solid.material.straintensor import (
    _strain_to_engineering,
    _strain_to_tensorial,
)
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix


def symmetric_3x3() -> np.ndarray:
    arr = np.random.rand(3, 3)
    return arr + arr.T


class TestTensor2x3(SolidMaterialTestCase):
    def setUp(self):
        self.frame = ReferenceFrame(dim=3)

    def test_single_from_matrix(self):
        arr = symmetric_3x3()
        tensor = Tensor2x3(arr.copy(), frame=self.frame)
        self.assertTrue(np.allclose(tensor.array, arr))
        # 11, 22, 33, 23, 13, 12
        expected = [arr[0, 0], arr[1, 1], arr[2, 2], arr[1, 2], arr[0, 2], arr[0, 1]]
        self.assertTrue(np.allclose(tensor.contracted_components(), expected))

    def test_single_from_voigt_vector(self):
        vector = np.random.rand(6)
        tensor = Tensor2x3(vector, frame=self.frame)
        self.assertEqual(tensor.array.shape, (3, 3))
        self.assertTrue(np.allclose(tensor.array, tensor.array.T))
        self.assertTrue(np.allclose(tensor.contracted_components(), vector))

    def test_bulk(self):
        self.assertEqual(
            Tensor2x3(np.random.rand(5, 6), frame=self.frame).array.shape, (5, 3, 3)
        )
        self.assertEqual(
            Tensor2x3(np.random.rand(5, 2, 6), frame=self.frame).array.shape,
            (5, 2, 3, 3),
        )
        self.assertEqual(
            Tensor2x3(np.random.rand(5, 3, 3), frame=self.frame).array.shape,
            (5, 3, 3),
        )

    def test_invalid_input(self):
        frame = self.frame
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor2x3, np.random.rand(4), frame=frame
        )
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor2x3, np.random.rand(5, 3), frame=frame
        )
        self.assertFailsProperly(
            ValueError, Tensor2x3, np.random.rand(5, 6), frame=frame, bulk=False
        )
        self.assertFailsProperly(
            ValueError, Tensor2x3, np.random.rand(5, 3, 3), frame=frame, bulk=False
        )
        self.assertFailsProperly(
            ValueError, Tensor2x3, np.random.rand(6), frame=frame, bulk=True
        )
        self.assertFailsProperly(
            ValueError, Tensor2x3, symmetric_3x3(), frame=frame, bulk=True
        )

    def test_verify_input(self):
        self.assertTrue(Tensor2x3._verify_input(np.zeros((3, 3))))
        self.assertTrue(Tensor2x3._verify_input(np.zeros((6,))))
        self.assertFalse(Tensor2x3._verify_input(np.zeros((2, 3, 3))))
        self.assertTrue(Tensor2x3._verify_input(np.zeros((2, 3, 3)), bulk=True))
        self.assertTrue(Tensor2x3._verify_input(np.zeros((2, 6)), bulk=True))
        self.assertFalse(Tensor2x3._verify_input(np.zeros((3, 3)), bulk=True))

    def test_transformation(self):
        arr = symmetric_3x3()
        source = ReferenceFrame(dim=3)
        target = source.rotate_new("Space", [0, 0, np.pi / 3], "123")
        tensor = Tensor2x3(arr.copy(), frame=source)

        # components in the target frame, compared to the classical formula
        dcm = source.dcm(target=target)
        expected = dcm @ arr @ dcm.T
        components = tensor.show(target)
        self.assertTrue(np.allclose(components, expected))
        self.assertTrue(np.allclose(tensor.transform_components(dcm), expected))

        # contracted components in another frame
        contracted = tensor.contracted_components(target)
        self.assertTrue(np.allclose(contracted[:3], np.diag(expected)))

        # rotating there and back gives the original components
        tensor.orient("Space", [0, 0, np.pi / 3], "123")
        tensor.orient("Space", [0, 0, -np.pi / 3], "123")
        self.assertTrue(np.allclose(tensor.array, arr))


class TestCauchyStressTensor(SolidMaterialTestCase):
    def test_transpose_returns_self(self):
        tensor = CauchyStressTensor(symmetric_3x3(), frame=ReferenceFrame(dim=3))
        self.assertIs(tensor.transpose(), tensor)


class TestSmallStrainTensor(SolidMaterialTestCase):
    def setUp(self):
        self.frame = ReferenceFrame(dim=3)

    def test_engineering_input(self):
        # engineering strains in the order 11, 22, 33, 23, 13, 12
        strains = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tensor = SmallStrainTensor(strains.copy(), frame=self.frame)
        # the tensorial shear strains are half of the engineering ones
        self.assertEqual(tensor.array[1, 2], 2.0)
        self.assertEqual(tensor.array[0, 2], 2.5)
        self.assertEqual(tensor.array[0, 1], 3.0)
        self.assertTrue(np.allclose(tensor.contracted_components(), strains))
        self.assertTrue(
            np.allclose(
                tensor.contracted_components(engineering=False),
                [1.0, 2.0, 3.0, 2.0, 2.5, 3.0],
            )
        )

    def test_tensorial_input(self):
        arr = symmetric_3x3()
        tensor = SmallStrainTensor(arr.copy(), frame=self.frame, tensorial=True)
        self.assertTrue(np.allclose(tensor.array, arr))
        self.assertIs(tensor.transpose(), tensor)

    def test_conversion_helpers(self):
        arr = symmetric_3x3()
        work = arr.copy()
        _strain_to_engineering(work)
        self.assertTrue(np.allclose(np.diag(work), np.diag(arr)))
        self.assertTrue(np.allclose(work[0, 1], 2 * arr[0, 1]))
        _strain_to_tensorial(work)
        self.assertTrue(np.allclose(work, arr))

        vector = np.random.rand(6)
        work = vector.copy()
        _strain_to_tensorial(work)
        self.assertTrue(np.allclose(work[3:], vector[3:] / 2))
        _strain_to_engineering(work)
        self.assertTrue(np.allclose(work, vector))


class TestElasticityTensorCreation(SolidMaterialTestCase):
    def setUp(self):
        self.frame = ReferenceFrame(dim=3)
        self.params = dict(E=210000.0, NU=0.3)
        self.hooke = elastic_stiffness_matrix(**self.params)

    def test_from_dictionary(self):
        tensor = ElasticityTensor(dict(self.params), frame=self.frame)
        self.assertTrue(np.allclose(tensor.contracted_components(), self.hooke))
        self.assertEqual(tensor.material_parameters, self.params)

    def test_from_keyword_arguments(self):
        tensor = ElasticityTensor(frame=self.frame, **self.params)
        self.assertTrue(np.allclose(tensor.elastic_stiffness_matrix(), self.hooke))
        # the parameters can be used to create a clone in the same frame
        params = tensor.material_parameters
        self.assertEqual({k: params[k] for k in self.params}, self.params)
        clone = ElasticityTensor(**params)
        self.assertTrue(np.allclose(clone.contracted_components(), self.hooke))

    def test_material_parameters_of_array_input(self):
        tensor = ElasticityTensor(self.hooke, frame=self.frame)
        self.assertEqual(tensor.material_parameters.shape, (3, 3, 3, 3))

    def test_yield_strength(self):
        tensor = ElasticityTensor(dict(self.params), frame=self.frame)
        self.assertEqual(tensor.yield_strength, np.inf)
        tensor.yield_strength = 355.0
        self.assertEqual(tensor.yield_strength, 355.0)
        self.assertEqual(tensor.material_parameters["yield_strength"], 355.0)
        tensor.yield_strength = None
        self.assertEqual(tensor.yield_strength, np.inf)

        tensor = ElasticityTensor(self.hooke, frame=self.frame)
        tensor.yield_strength = 235.0
        self.assertEqual(tensor.yield_strength, 235.0)

    def test_bulk(self):
        bulk = np.stack([self.hooke, self.hooke])
        tensor = ElasticityTensor(bulk, frame=self.frame)
        self.assertEqual(tensor.array.shape, (2, 3, 3, 3, 3))
        tensor = ElasticityTensor(tensor.array, frame=self.frame)
        self.assertEqual(tensor.array.shape, (2, 3, 3, 3, 3))

    def test_full_4d_input(self):
        array = ElasticityTensor(self.hooke, frame=self.frame).array
        tensor = ElasticityTensor(array, frame=self.frame)
        self.assertTrue(np.allclose(tensor.contracted_components(), self.hooke))
        self.assertIs(tensor.transpose(), tensor)

    def test_verify_input(self):
        self.assertTrue(ElasticityTensor._verify_input(np.zeros((6, 6))))
        self.assertTrue(ElasticityTensor._verify_input(np.zeros((3, 3, 3, 3))))
        self.assertTrue(ElasticityTensor._verify_input(np.zeros((2, 6, 6)), bulk=True))
        self.assertTrue(
            ElasticityTensor._verify_input(np.zeros((2, 3, 3, 3, 3)), bulk=True)
        )
        self.assertFalse(ElasticityTensor._verify_input(np.zeros((6, 6)), bulk=True))

    def test_invalid_input(self):
        frame = self.frame
        bulk = np.stack([self.hooke, self.hooke])
        array = ElasticityTensor(self.hooke, frame=frame).array
        self.assertFailsProperly(
            ValueError, ElasticityTensor, bulk, frame=frame, bulk=False
        )
        self.assertFailsProperly(
            ValueError, ElasticityTensor, self.hooke, frame=frame, bulk=True
        )
        self.assertFailsProperly(
            ValueError, ElasticityTensor, array[None, ...], frame=frame, bulk=False
        )
        self.assertFailsProperly(
            ValueError, ElasticityTensor, array, frame=frame, bulk=True
        )
        self.assertFailsProperly(
            TensorShapeMismatchError, ElasticityTensor, np.zeros((4, 4)), frame=frame
        )
        self.assertFailsProperly(
            TensorShapeMismatchError, ElasticityTensor, np.zeros(6), frame=frame
        )
        self.assertFailsProperly(
            TensorShapeMismatchError, ElasticityTensor, np.zeros((3, 3)), frame=frame
        )

    def test_invalid_number_of_components(self):
        tensor = ElasticityTensor(self.hooke, frame=self.frame)
        self.assertFailsProperly(ValueError, tensor.calculate_stresses, np.zeros(5))
        self.assertFailsProperly(ValueError, tensor.calculate_strains, np.zeros(5))

    def test_stresses_and_strains_are_inverse(self):
        tensor = ElasticityTensor(self.hooke, frame=self.frame)
        strains = np.random.rand(10, 6)
        stresses = tensor.calculate_stresses(strains)
        self.assertTrue(np.allclose(tensor.calculate_strains(stresses), strains))
        self.assertTrue(np.allclose(tensor.calculate_strains(stresses[0]), strains[0]))


if __name__ == "__main__":
    unittest.main()
