import unittest

import numpy as np
import xarray as xr

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
    HuberMisesHenckyFailureCriterion,
    HuberMisesHenckyFailureCriterion_SP,
    HuberMisesHenckyFailureCriterion_M,
)
from sigmaepsilon.solid.material.evaluator import FunctionEvaluator
from sigmaepsilon.solid.material.utils.hmh import HMH_3d_v

CUDA_DEVICES = ["cuda", "cuda-guv", "cuda-kernel", "cuda-cp"]


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


class TestHMHUtilization(SolidMaterialTestCase):
    def setUp(self):
        self.yield_strength = 2.0
        self.obj = HuberMisesHenckyFailureCriterion(yield_strength=self.yield_strength)
        # uniaxial stress and pure shear at the yield strength, in the order
        # s11, s22, s33, s23, s13, s12
        stresses = np.zeros((2, 6))
        stresses[0, 0] = self.yield_strength
        stresses[1, 5] = self.yield_strength / np.sqrt(3)
        self.stresses = stresses

    def test_known_values(self):
        obj, stresses = self.obj, self.stresses
        self.assertTrue(np.allclose(obj.utilization(stresses=stresses), 1.0))
        self.assertTrue(np.allclose(obj(stresses=stresses), 1.0))
        self.assertTrue(
            np.allclose(obj.utilization(stresses=tuple(stresses.T.copy())), 1.0)
        )
        self.assertTrue(
            np.allclose(obj.utilization(stresses=xr.DataArray(stresses)), 1.0)
        )
        self.assertEqual(obj.utilization(stresses=stresses, squeeze=False).shape, (2,))

    def test_params(self):
        obj = HuberMisesHenckyFailureCriterion()
        self.assertEqual(obj.number_of_strength_parameters, 1)
        obj.params = [3.0]
        self.assertEqual(obj.params, [3.0])
        obj.params = 4.0
        self.assertEqual(obj.yield_strength, 4.0)
        self.assertFailsProperly(ValueError, setattr, obj, "params", [1.0, 2.0])
        self.assertFailsProperly(TypeError, setattr, obj, "params", "a")
        self.assertEqual(HuberMisesHenckyFailureCriterion(params=[5.0]).yield_strength, 5.0)

    def test_invalid_device(self):
        stresses = self.stresses
        for s in (stresses, tuple(stresses.T.copy())):
            self.assertFailsProperly(
                NotImplementedError, self.obj.utilization, stresses=s, device="tpu"
            )

    def test_single_evaluator(self):
        # a vectorized implementation is used for every input
        evaluator = FunctionEvaluator(HMH_3d_v, vectorized=True)
        obj = HuberMisesHenckyFailureCriterion(self.yield_strength, evaluator=evaluator)
        self.assertTrue(np.allclose(obj.utilization(stresses=self.stresses), 1.0))
        self.assertTrue(
            np.allclose(obj.utilization(stresses=tuple(self.stresses.T.copy())), 1.0)
        )

    def test_device_dispatch(self):
        # the CPU implementation stands in for the CUDA ones to test the dispatch
        evaluator = FunctionEvaluator(HMH_3d_v, vectorized=True, cuda=True)
        self.assertTrue(evaluator.is_cuda and evaluator.is_vectorized)
        self.assertFalse(evaluator.is_bulk)
        evaluators = {device: evaluator for device in CUDA_DEVICES}
        obj = HuberMisesHenckyFailureCriterion(self.yield_strength, evaluator=evaluators)
        for device in CUDA_DEVICES:
            stresses = tuple(self.stresses.T.copy())
            utilization = obj.utilization(stresses=stresses, device=device)
            self.assertTrue(np.allclose(utilization, 1.0))
        for device in ["cuda", "cuda-cp"]:
            utilization = obj.utilization(stresses=self.stresses, device=device)
            self.assertTrue(np.allclose(utilization, 1.0))

    def test_fit_recovers_yield_strength(self):
        stresses = np.concatenate([self.stresses, 2 * self.stresses])
        utilization = np.array([1.0, 1.0, 2.0, 2.0])
        obj = HuberMisesHenckyFailureCriterion()
        params = obj.fit(stresses, utilization, method="Nelder-Mead", x0=[5.0], tol=1e-8)
        self.assertAlmostEqual(params[0], self.yield_strength, places=4)
        params = obj.fit(stresses, utilization, method="bga", n_iter=50)
        self.assertAlmostEqual(params[0], self.yield_strength, delta=0.1)


if __name__ == "__main__":
    unittest.main()
