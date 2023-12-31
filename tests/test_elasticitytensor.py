import unittest
import numpy as np

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import ElasticityTensor
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix


def membrane_part(Q_in: np.ndarray):
    Q = np.zeros((3, 3), dtype=float)
    Q[0, 0] = Q_in[0, 0]
    Q[0, 1] = Q_in[0, 1]
    Q[1, 0] = Q_in[1, 0]
    Q[1, 1] = Q_in[1, 1]
    Q[0, 2] = Q_in[0, 5]
    Q[1, 2] = Q_in[1, 5]
    Q[2, 0] = Q_in[0, 5]
    Q[2, 1] = Q_in[1, 5]
    Q[2, 2] = Q_in[5, 5]
    return Q


class TestElasticityTensor(SolidMaterialTestCase):
    def test_1(self):
        alpha = 45.0 * np.pi / 180
        
        # create a material stiffness matrix in the 123 frame, which is bound
        # to the anatomical directions of the material
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
        
        # create a 4d tensor from this matrix and check if the 2d representation of it
        # coincides with the matrix
        frame_x = ReferenceFrame(dim=3)
        tensor = ElasticityTensor(hooke_123, frame=frame_x, tensorial=False)
        cc = tensor.contracted_components()
        self.assertLess(np.sum(np.abs(hooke_123 - cc)), 1e-8)
        
        # define the same material in a rotated frame and calculate the components in
        # the original frame
        frame_x = ReferenceFrame(dim=3)
        frame_y = frame_x.rotate_new("Space", [0, 0, alpha], "123")
        tensor = ElasticityTensor(hooke_123, frame=frame_y)
        hooke_xyz = tensor.contracted_components(target=frame_x)
        hooke_xyz[np.where(np.abs(hooke_xyz) < 1e-5)] = 0
        
        # calculate the components of the membrane part of the original matrix in the
        # rotated frame
        Q_12 = membrane_part(hooke_123)
        S = np.sin(alpha)
        C = np.cos(alpha)
        Q_xy = np.zeros((3, 3), dtype=float)
        Q_xy[0, 0] = (
            Q_12[0, 0] * C**4
            + 2 * (Q_12[0, 1] + 2 * Q_12[2, 2]) * S**2 * C**2
            + Q_12[1, 1] * S**4
        )
        Q_xy[0, 1] = (Q_12[0, 0] + Q_12[1, 1] - 4 * Q_12[2, 2]) * S**2 * C**2 + Q_12[
            0, 1
        ] * (S**4 + C**4)
        Q_xy[1, 1] = (
            Q_12[0, 0] * S**4
            + 2 * (Q_12[0, 1] + 2 * Q_12[2, 2]) * S**2 * C**2
            + Q_12[1, 1] * C**4
        )
        Q_xy[0, 2] = (Q_12[0, 0] - Q_12[0, 1] - 2 * Q_12[2, 2]) * S * C**3 + (
            Q_12[0, 1] - Q_12[1, 1] + 2 * Q_12[2, 2]
        ) * S**3 * C
        Q_xy[1, 2] = (Q_12[0, 0] - Q_12[0, 1] - 2 * Q_12[2, 2]) * S**3 * C + (
            Q_12[0, 1] - Q_12[1, 1] + 2 * Q_12[2, 2]
        ) * S * C**3
        Q_xy[2, 2] = (
            Q_12[0, 0] + Q_12[1, 1] - 2 * Q_12[0, 1] - 2 * Q_12[2, 2]
        ) * S**2 * C**2 + Q_12[2, 2] * (S**4 + C**4)
        Q_xy[1, 0] = Q_xy[0, 1]
        Q_xy[2, 0] = Q_xy[0, 2]
        Q_xy[2, 1] = Q_xy[1, 2]
        Q_xy[np.where(np.abs(Q_xy) < 1e-5)] = 0
        
        # check if it is the same as with the tensor
        self.assertLess(np.sum(np.abs(membrane_part(hooke_xyz) - Q_xy)), 1e-8)
        
    def test_2(self):
        matrix = elastic_stiffness_matrix(
            E1=2100.0,
            E2=210.0,
            G13=3000.0,
            NU12=0.2,
            NU23=0.02,
            isoplane="23",
            verify=True,
        )
        self.assertValidMaterial(matrix)
                
        angles = np.array([0, 0, np.random.rand() * np.pi])
        frame_x = ReferenceFrame(dim=3)
        tensor = ElasticityTensor(matrix, frame=frame_x, tensorial=False)
        tensor.orient("Space", angles, "123")
        tensor.orient("Space", -angles, "123")
        _matrix = tensor.contracted_components()
        self.assertLess(np.sum(np.abs(matrix - _matrix)), 1e-8)
        
        angles = np.array([0, np.random.rand() * np.pi, 0])
        frame_x = ReferenceFrame(dim=3)
        tensor = ElasticityTensor(matrix, frame=frame_x, tensorial=False)
        tensor.orient("Space", angles, "123")
        tensor.orient("Space", -angles, "123")
        _matrix = tensor.contracted_components()
        self.assertLess(np.sum(np.abs(matrix - _matrix)), 1e-8)
        
        angles = np.array([np.random.rand() * np.pi, 0, 0])
        frame_x = ReferenceFrame(dim=3)
        tensor = ElasticityTensor(matrix, frame=frame_x, tensorial=False)
        tensor.orient("Space", angles, "123")
        tensor.orient("Space", -angles, "123")
        _matrix = tensor.contracted_components()
        self.assertLess(np.sum(np.abs(matrix - _matrix)), 1e-8)
    
    def test_isotropic(self):        
        hooke = elastic_stiffness_matrix(E=210000, NU=0.3)
        yield_strength=355.0

        self.assertValidMaterial(hooke)

        # create a 4d tensor from this matrix and check if the 2d representation of it
        # coincides with the matrix
        frame = ReferenceFrame(dim=3)
        tensor = ElasticityTensor(hooke, frame=frame, tensorial=False, yield_strength=yield_strength)
        
        strains = tensor.calculate_strains(np.array([yield_strength, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.isclose(strains[1], strains[2]))
        
        self.assertEqual(tensor.calculate_stresses(np.random.rand(6)).shape, (6,))
        self.assertEqual(tensor.calculate_stresses(np.random.rand(10, 6)).shape, (10, 6))
        
        strains = tensor.calculate_strains(np.array([yield_strength, 0.0, 0.0, 0.0, 0.0, 0.0]))
        util = tensor.utilization(strains) * 100
        self.assertTrue(np.isclose(util, 100.0))
        
        strains = tensor.calculate_strains(np.eye(6) * yield_strength)
        utils = tensor.utilization(strains=strains) * 100
        self.assertTrue(np.isclose(utils[0], 100.0))
        self.assertTrue(np.isclose(utils[1], 100.0))
        self.assertTrue(np.isclose(utils[2], 100.0))
        
        value_a = tensor.calculate_equivalent_stress(np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0]))
        value_b = tensor.calculate_equivalent_stress(np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.isclose(value_a, value_b))
        
        # simple evaluations
        tensor.utilization(2*np.random.rand(10, 6)/1000)
        tensor.utilization(*(2*np.random.rand(10)/1000 for _ in range(6)))
        tensor.utilization(*(2*np.random.rand(10)/1000 for _ in range(6))) 

if __name__ == "__main__":
    unittest.main()
