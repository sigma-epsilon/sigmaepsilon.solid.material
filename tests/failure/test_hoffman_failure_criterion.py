import unittest

import numpy as np

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
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


class TestHoffmanFailureCriterionUtils(SolidMaterialTestCase):
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

        Hoffman_failure_criterion_principal_form(*((np.ones(3),) + (1,) * 14))

    def test_Hoffman_failure_criterion_principal_form_PS(self):
        Hoffman_failure_criterion_principal_form_PS(*(1,) * 14)
        Hoffman_failure_criterion_principal_form_PS(*(np.ones(2),) * 14)
        Hoffman_failure_criterion_principal_form_PS(*((np.ones(3),) + (1,) * 13))

    def test_Hoffman_failure_criterion_principal_form_M(self):
        Hoffman_failure_criterion_principal_form_M(*(1,) * 10)
        Hoffman_failure_criterion_principal_form_M(*(np.ones(2),) * 10)
        Hoffman_failure_criterion_principal_form_M(*((np.ones(3),) + (1,) * 9))


class TestHoffmanFailureCriterion(SolidMaterialTestCase):
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


class TestHoffmanFitting3d(SolidMaterialTestCase):
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

        failure_obj = HoffmanFailureCriterion()

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
            x0=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.2, 0.2, 0.2],
        )
        self.failure_obj.params = params
        self._predict()

    def test_fit_BGA(self):
        params = self.failure_obj.fit(
            self.inputs,
            self.outputs,
            solver_params=dict(nPop=10, length=6),
            penalty=1e12,
            tol=0.1,
            n_iter=1,
            ranges=[[-10, 10] for _ in range(9)],
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
            x0=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.2, 0.2, 0.2],
        )
        self.failure_obj.params = params
        self._predict()

    def _predict(self):
        prediction = self.failure_obj.utilization(stresses=self.inputs)
        max_error = np.max(np.abs(prediction - self.outputs))
        total_error = np.sum(np.sqrt((prediction - self.outputs) ** 2))
        return prediction, max_error, total_error
    

class TestHoffmanFittingMembrane(SolidMaterialTestCase):
    def setUp(self):
        inputs = [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [-1, 0.2, 0],
            [0.2, 1, 0],
            [-1, 0, 0.2],
        ]
        inputs = np.array(inputs, dtype=float)
        outputs = np.ones(len(inputs))

        failure_obj = HoffmanFailureCriterion_M()

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
            x0=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.2],
        )
        self.failure_obj.params = params
        self._predict()

    def test_fit_BGA(self):
        params = self.failure_obj.fit(
            self.inputs,
            self.outputs,
            solver_params=dict(nPop=10, length=12),
            penalty=1e12,
            tol=0.1,
            n_iter=1,
            ranges=[[-10, 10] for _ in range(7)],
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
            x0=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.2],
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
