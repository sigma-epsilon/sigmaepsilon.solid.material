from typing import Union, Optional
from numbers import Number
import warnings

import numpy as np
from numpy import ndarray, ascontiguousarray as ascont
from numpy.linalg import inv

from sigmaepsilon.math import atleast2d

from ..proto import MaterialLike, FailureLike
from .mindlinshell import MindlinShellSection, MindlinShellLayer
from ..utils.mindlin import (
    _get_shell_material_stiffness_matrix,
    shell_rotation_matrix,
)
from ..warnings import SigmaEpsilonMaterialWarning
from ..enums import MaterialModelType
from ..linearelasticmaterial import LinearElasticMaterial
from ..failure import HuberMisesHenckyFailureCriterion_M

__all__ = ["MembraneSection"]


class MembraneLayer(MindlinShellLayer):
    """
    A class for layers of a membrane.
    """

    def material_elastic_stiffness_matrix(
        self, shape: Optional[Union[str, None]] = "full"
    ) -> ndarray:
        """
        Returns and stores the transformed material stiffness matrix.
        """
        Cm = _get_shell_material_stiffness_matrix(self.material_stiffness_matrix)
        T = self.rotation_matrix(shape=shape)

        if isinstance(shape, str):
            if shape.lower() == "full":
                full_shape = True
            elif shape.lower() == "contracted":
                full_shape = False
            else:
                raise ValueError(f"Invalid value '{shape}' for parameter 'shape'")

        R = (
            np.diag([1.0, 1.0, 2.0, 2.0, 2.0])
            if full_shape
            else np.diag([1.0, 1.0, 2.0])
        )

        if not full_shape:
            Cm = Cm[:3, :3]

        return inv(T) @ Cm @ R @ T @ inv(R)

    def rotation_matrix(self, shape: Optional[Union[str, None]] = "full") -> ndarray:
        """
        Returns the transformation matrix of the layer.
        """
        if isinstance(shape, str):
            if shape.lower() == "full":
                return shell_rotation_matrix(self.angle, rad=False)
            elif shape.lower() == "contracted":
                return ascont(shell_rotation_matrix(self.angle, rad=False)[:3, :3])
            else:
                raise ValueError(f"Invalid value '{shape}' for parameter 'shape'")
        raise NotImplementedError

    def stresses(self, *, strains: ndarray, out: ndarray = None, **__) -> ndarray:
        C = self.material_elastic_stiffness_matrix()
        num_component = C.shape[0]

        strains = atleast2d(strains, front=True)
        num_data = strains.shape[0]

        if out is None:
            out = np.zeros((num_data, num_component), dtype=float)

        out[:, :3] = (C[:3, :3] @ strains[:, :3].T).T

        return out


class MembraneSection(MindlinShellSection):
    """
    A class for membranes.

    It is a subclass of class::`sigmaepsilon.solid.material.surface.SurfaceSection`, and usage is
    almost identical, see there for more details.

    Examples
    --------
    >>> from sigmaepsilon.solid.material import MembraneSection as Section
    >>> from sigmaepsilon.math.linalg import ReferenceFrame
    >>> from sigmaepsilon.solid.material import ElasticityTensor
    >>> from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix
    >>> ...
    >>> E = 2890.0
    >>> nu = 0.2
    >>> yield_strength = 2.0
    >>> thickness = 25.0
    >>> ...
    >>> hooke = elastic_stiffness_matrix(E=E, NU=nu)
    >>> frame = ReferenceFrame(dim=3)
    >>> tensor = ElasticityTensor(
    ...    hooke, frame=frame, tensorial=False, yield_strength=yield_strength
    ... )
    >>> ...
    >>> section = Section(
    ...     layers=[
    ...         Section.Layer(material=tensor, thickness=thickness / 3),
    ...         Section.Layer(material=tensor, thickness=thickness / 3),
    ...         Section.Layer(material=tensor, thickness=thickness / 3),
    ...     ]
    ... )
    >>> section.elastic_stiffness_matrix().shape
    (3, 3)
    """

    layer_class = MembraneLayer
    model_type = MaterialModelType.MEMBRANE
    failure_class: FailureLike = HuberMisesHenckyFailureCriterion_M
    number_of_stress_components: int = 3

    def __init__(self, *args, assume_regular: bool = False, **kwargs):
        self._assume_regular = assume_regular
        return super().__init__(*args, **kwargs)

    @MindlinShellSection.eccentricity.setter
    def eccentricity(self, _: Number) -> None:
        raise Exception(
            "Membranes can't have eccentricity, consider using a shell model instead."
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
                SigmaEpsilonMaterialWarning,
            )

        self._ABDS = ascont(A)
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS
