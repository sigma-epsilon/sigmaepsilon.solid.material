from numpy import ndarray

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math.logical import isposdef, issymmetric


class SolidMaterialTestCase(SigmaEpsilonTestCase):
    def assertValidMaterial(self, D: ndarray, stol: float = 1e-8):
        self.assertTrue(isposdef(D), "The stiffness matrix is not positive definite.")
        self.assertTrue(issymmetric(D, stol), "The stiffness matrix is not symmetric.")
