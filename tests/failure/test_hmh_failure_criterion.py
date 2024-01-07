import unittest

import numpy as np

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
    HuberMisesHenckyFailureCriterion,
    HuberMisesHenckyFailureCriterion_SP,
    HuberMisesHenckyFailureCriterion_M,
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
        
class TestHMHFitting3d(SolidMaterialTestCase):
    def setUp(self):
        inputs = [
            [-1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 1],
            [-1, 0.2, 0, 0, 0, 0],
            [0.2, 1, 0, 0, 0, 0],
            [-1, 0, 0.2, 0, 0, 0],
        ]
        inputs = np.array(inputs, dtype=float)
        outputs = np.ones(len(inputs))

        failure_obj = HuberMisesHenckyFailureCriterion()

        self.inputs = inputs
        self.outputs = outputs
        self.failure_obj = failure_obj

    def test_fit_invalid_method_raises_TypeError(self):
        self.assertRaises(
            TypeError, self.failure_obj.fit, self.inputs, self.outputs, method=1
        )

    def test_fit_auto(self):
        params = self.failure_obj.fit(
            self.inputs,
            self.outputs,
        )
        self.failure_obj.params = params
        self._predict()

    def test_fit_BGA(self):
        params = self.failure_obj.fit(
            self.inputs,
            self.outputs,
            solver_params=dict(nPop=100, length=12),
            penalty=1e12,
            tol=0.1,
            n_iter=100,
            ranges=[[-10, 10]],
            method="bga",
        )
        self.failure_obj.params = params
        self._predict()

    def test_fit_NM(self):
        params = self.failure_obj.fit(
            self.inputs,
            self.outputs,
            penalty=1e12,
            tol=0.1,
            method="Nelder-Mead",
            x0=[1.0],
        )
        self.failure_obj.params = params
        self._predict()

    def _predict(self):
        prediction = self.failure_obj.utilization(stresses=self.inputs)
        max_error = np.max(np.abs(prediction - self.outputs))
        total_error = np.sum(np.sqrt((prediction - self.outputs) ** 2))
        return prediction, max_error, total_error


if __name__ == "__main__":
    unittest.main()
