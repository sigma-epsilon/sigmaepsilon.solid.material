import unittest

import numpy as np

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
    HuberMisesHenckyFailureCriterion,
    HuberMisesHenckyFailureCriterion_SP,
    HuberMisesHenckyFailureCriterion_M,
    HoffmanFailureCriterion,
    HoffmanFailureCriterion_SP,
    HoffmanFailureCriterion_M,
)
from sigmaepsilon.solid.material.utils.hoffman import (
    Hoffman_failure_criterion_standard_form,
    Hoffman_failure_criterion_principal_form,
    Hoffman_failure_criterion_principal_form_PS,
    Hoffman_failure_criterion_principal_form_M,
)


class TestHMHFailureCriterion(SolidMaterialTestCase):
    def test_general_behaviour(self):
        obj = HuberMisesHenckyFailureCriterion(yield_strength=1.0)

        self.assertTrue(np.isclose(obj.yield_strength, 1.0))

        obj.utilization(stresses=np.random.rand(6))
        obj.utilization(stresses=np.random.rand(2, 6))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(6)))

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )

    def test_general_behaviour_M(self):
        obj = HuberMisesHenckyFailureCriterion_M(yield_strength=1.0)

        self.assertTrue(np.isclose(obj.yield_strength, 1.0))

        obj.utilization(stresses=np.random.rand(3))
        obj.utilization(stresses=np.random.rand(2, 3))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(3)))

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )

    def test_general_behaviour_SP(self):
        obj = HuberMisesHenckyFailureCriterion_SP(yield_strength=1.0)

        self.assertTrue(np.isclose(obj.yield_strength, 1.0))

        obj.utilization(stresses=np.random.rand(5))
        obj.utilization(stresses=np.random.rand(2, 5))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(5)))

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )


class TestHoffmanFailureCriterion(SolidMaterialTestCase):
    def test_Hoffman_failure_criterion_principal_form(self):
        self.assertTrue(
            np.allclose(Hoffman_failure_criterion_standard_form(*(1,) * 15), 6.0)
        )
        self.assertTrue(
            np.allclose(
                Hoffman_failure_criterion_standard_form(*(np.ones(2),) * 15), [6.0, 6.0]
            )
        )
        self.assertTrue(
            np.allclose(
                Hoffman_failure_criterion_standard_form(*((np.ones(3),) + (1,) * 14)),
                [6.0, 6.0, 6.0],
            )
        )
        
    def test_Hoffman_failure_criterion_principal_form_PS(self):
        Hoffman_failure_criterion_principal_form_PS(*(1,) * 14)
        Hoffman_failure_criterion_principal_form_PS(*(np.ones(2),) * 14)
        Hoffman_failure_criterion_principal_form_PS(*((np.ones(3),) + (1,) * 13))
        
    def test_Hoffman_failure_criterion_principal_form_M(self):
        Hoffman_failure_criterion_principal_form_M(*(1,) * 10)
        Hoffman_failure_criterion_principal_form_M(*(np.ones(2),) * 10)
        Hoffman_failure_criterion_principal_form_M(*((np.ones(3),) + (1,) * 9))

    def test_general_behaviour(self):
        obj = HoffmanFailureCriterion(params=[1.0 for _ in range(9)])

        obj.utilization(stresses=np.random.rand(6))
        obj.utilization(stresses=np.random.rand(2, 6))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(6)))
        
        self.assertEqual(obj.number_of_strength_parameters, 9)

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )

    def test_general_behaviour_SP(self):
        obj = HoffmanFailureCriterion_SP(params=[1.0 for _ in range(9)])

        obj.utilization(stresses=np.random.rand(5))
        obj.utilization(stresses=np.random.rand(2, 5))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(5)))
        
        self.assertEqual(obj.number_of_strength_parameters, 9)

        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 8)
        )

    def test_general_behaviour_M(self):
        
        obj = HoffmanFailureCriterion_M(params=[1.0 for _ in range(7)])

        obj.utilization(stresses=np.random.rand(3))
        obj.utilization(stresses=np.random.rand(2, 3))
        obj.utilization(stresses=tuple(np.random.rand(2) for _ in range(3)))

        self.assertEqual(obj.number_of_strength_parameters, 7)
        
        self.assertFailsProperly(TypeError, obj.utilization, stresses="_")
        self.assertFailsProperly(
            ValueError, obj.utilization, stresses=np.random.rand(2, 5)
        )


if __name__ == "__main__":
    unittest.main()
