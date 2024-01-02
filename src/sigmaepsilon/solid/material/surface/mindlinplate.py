from numbers import Number
import warnings

from numpy import ndarray, ascontiguousarray as ascont
import numpy as np

from sigmaepsilon.math import atleast2d

from .mindlinshell import MindlinShellLayer, MindlinShellSection
from ..utils.mindlin import z_to_shear_factors
from ..warnings import SigmaEpsilonMaterialWarning
from ..enums import MaterialModelType

__all__ = ["MindlinPlateSection"]

class MindlinPlateLayer(MindlinShellLayer):
    """
    A class for layers of a membrane.
    """
    
    def stresses(self, *, strains:ndarray, stresses:ndarray, z:float=None, out:ndarray=None) -> ndarray:
        C = self.material_elastic_stiffness_matrix()
        num_component = C.shape[0]
        
        strains = atleast2d(strains, front=True)
        num_data = strains.shape[0]
        
        if out is None:
            out = np.zeros((num_data, num_component), dtype=float)
        
        out[:, :3] = (C[:3, :3] @ (z * strains[:, :3]).T).T

        sfx, sfy = z_to_shear_factors(z, self._sfx, self._sfy)
        out[:, 3] = sfx * stresses[:, -2]
        out[:, 4] = sfy * stresses[:, -1]
        
        return out

class MindlinPlateSection(MindlinShellSection):
    """
    A class for Uflyand-Mindlin plates.

    Example
    -------
    >>> from sigmaepsilon.solid.material import MindlinPlateSection as Section
    >>> from sigmaepsilon.math.linalg import ReferenceFrame
    >>> from sigmaepsilon.solid.material import ElasticityTensor
    >>> from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix
    >>> import numpy as np
    >>> yield_strength = 355.0
    >>> hooke = elastic_stiffness_matrix(E=210000, NU=0.3)
    >>> frame = ReferenceFrame(dim=3)
    >>> tensor = ElasticityTensor(
    ...    hooke, frame=frame, tensorial=False, yield_strength=yield_strength
    ... )
    >>> section = Section(
    ...     layers=[
    ...         Section.Layer(material=tensor, thickness=0.1),
    ...         Section.Layer(material=tensor, thickness=0.1),
    ...     ]
    ... )
    >>> section.elastic_stiffness_matrix().shape
    (5, 5)

    An instance of this class can be used for the postprocessing of one or several
    evaluations at once.

    The following call returns in an array with a shape of (2, 3). This is because 'z'
    was not specified and there are 2 layers with 3 points of evaluation.

    >>> section.calculate_equivalent_stress(strains=np.array([1, 0, 0, 0, 0])).shape
    (2, 3)

    The following call returns an array of shape (10, 2, 3) because feed it with 10
    evaluations (for more than one set of strains).

    >>> section.calculate_equivalent_stress(strains=2 * np.random.rand(10, 5) / 1000).shape
    (10, 2, 3)
    """

    model_type = MaterialModelType.PLATE_UFLYAND_MINDLIN
    layer_class = MindlinPlateLayer

    @MindlinShellSection.eccentricity.setter
    def eccentricity(self, _: Number) -> None:
        raise Exception(
            "Plates can't have eccentricity, consider using a shell instead."
        )

    def elastic_stiffness_matrix(self, tol: Number = 1e-8) -> ndarray:
        """
        Assembles and returns the stiffness matrix.
        """
        self._set_layers()
        ABDS = np.zeros(self.layer_class.__shape__)
        self._elastic_stiffness_matrix(ABDS)

        B = ABDS[:3, 3:6]
        sumB = np.sum(np.abs(B))
        if sumB > tol:
            warnings.warn(
                "It seems that handling this section as a plate only would result in "
                "missing out on the effect of bending-extension coupling. "
                "It is suggested to use a shell instead.",
                SigmaEpsilonMaterialWarning,
            )

        self._ABDS = ascont(ABDS[3:, 3:])
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS
