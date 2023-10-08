from numbers import Number
import warnings

import numpy as np
from numpy import ndarray, ascontiguousarray as ascont

from .mindlin import MindlinShellSection
from ..warnings import SigmaEpsilonMaterialWarning

__all__ = ["MembraneSection"]


class MembraneSection(MindlinShellSection):
    """
    A class for membranes.
    """

    @MindlinShellSection.eccentricity.setter
    def eccentricity(self, _: Number) -> None:
        raise Exception(
            "Membranes can't have eccentricity, consider using a shell instead."
        )

    def elastic_stiffness_matrix(self, tol: Number = 1e-8) -> ndarray:
        """
        Assembles and returns the stiffness matrix.
        """
        self._set_layers()
        ABDS = np.zeros(self.layer_class.__shape__)
        self._elastic_stiffness_matrix(ABDS)

        A = ABDS[:3, :3]
        B = ABDS[:3, 3:6]
        sumB = np.sum(np.abs(B))
        if sumB > tol:
            warnings.warn(
                "It seems that handling this section as a membrane only would result in "
                "missing out on the effect of bending-extension coupling. "
                "It is suggested to use a shell instead.",
                SigmaEpsilonMaterialWarning
            )

        self._ABDS = ascont(A)
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS
