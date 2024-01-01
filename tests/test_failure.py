import unittest

import numpy as np

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import HuberMisesHenckyFailureModel, HuberMisesHenckyFailureModel_SP, HuberMisesHenckyFailureModel_M


class TestHMHFailureModel(SolidMaterialTestCase):
    def test_general_behaviour(self):
        obj = HuberMisesHenckyFailureModel(yield_strength=1.0)

        self.assertTrue(np.isclose(obj.yield_strength, 1.0))

        obj.utilization(stresses=np.random.rand(6))
        obj.utilization(stresses=np.random.rand(2, 6))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(6)))

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )
        
    def test_general_behaviour_M(self):
        obj = HuberMisesHenckyFailureModel_M(yield_strength=1.0)

        self.assertTrue(np.isclose(obj.yield_strength, 1.0))

        obj.utilization(stresses=np.random.rand(3))
        obj.utilization(stresses=np.random.rand(2, 3))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(3)))

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )
        
    def test_general_behaviour_SP(self):
        obj = HuberMisesHenckyFailureModel_SP(yield_strength=1.0)

        self.assertTrue(np.isclose(obj.yield_strength, 1.0))

        obj.utilization(stresses=np.random.rand(5))
        obj.utilization(stresses=np.random.rand(2, 5))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(5)))

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )


if __name__ == "__main__":
    unittest.main()
