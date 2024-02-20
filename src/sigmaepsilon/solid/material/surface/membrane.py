from typing import Union, Optional, Tuple
from numbers import Number
import warnings

import numpy as np
from numpy import ndarray, ascontiguousarray as ascont
from numpy.linalg import inv, eigh
import xarray as xr

from sigmaepsilon.math import atleast2d

from ..proto import FailureLike
from .mindlinshell import MindlinShellSection, MindlinShellLayer
from ..utils.mindlin import (
    _get_shell_material_stiffness_matrix,
    shell_rotation_matrix,
)
from ..utils.principal import (
    principal_stress_angle_2d,
    max_principal_stress_2d,
    min_principal_stress_2d,
)
from ..warnings import SigmaEpsilonMaterialWarning
from ..enums import MaterialModelType
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
    material_stress_components = [
        "SXX",
        "SYY",
        "SXY",
    ]

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

    def principal_material_stress_angles(self, *args, **kwargs) -> xr.DataArray:
        """
        Returns principal material stress angles valuated at given heights
        from the reference surface. Calling the function is exactly the same as for
        `calculate_stresses`, see there for the details. Returns an `xarray.DataArray`
        instance.
        """
        stresses = self.calculate_stresses(*args, **kwargs)
        sxx, syy, sxy = (
            stresses.sel(component="SXX").values,
            stresses.sel(component="SYY").values,
            stresses.sel(component="SXY").values,
        )
        theta = principal_stress_angle_2d(sxx, syy, sxy)

        result_np = result = theta
        if len(result_np.shape) == 3:
            num_data, num_layers, num_point_per_layer = result_np.shape[:3]
            coords = [
                np.arange(num_data),
                np.arange(num_layers),
                np.arange(num_point_per_layer),
            ]
            dims = ["index", "layer", "point"]
            result = xr.DataArray(result_np, coords=coords, dims=dims)
        elif len(result_np.shape) == 2:
            num_data, num_z = result_np.shape[:2]
            coords = [np.arange(num_data), np.arange(num_z)]
            dims = ["index", "point"]
            result = xr.DataArray(result_np, coords=coords, dims=dims)
        else:
            raise NotImplementedError
        return result

    def principal_material_stresses(self, *args, **kwargs) -> xr.DataArray:
        """
        Returns principal material stresses valuated at given heights
        from the reference surface. Calling the function is exactly the same as for
        `calculate_stresses`, see there for the details. Returns an `xarray.DataArray`
        instance.
        """
        stresses = self.calculate_stresses(*args, **kwargs)
        sxx, syy, sxy = (
            stresses.sel(component="SXX").values,
            stresses.sel(component="SYY").values,
            stresses.sel(component="SXY").values,
        )
        s_1 = max_principal_stress_2d(sxx, syy, sxy)
        s_2 = min_principal_stress_2d(sxx, syy, sxy)

        result_np = result = np.stack([s_1, s_2], axis=-1)
        if len(result_np.shape) == 4:
            num_data, num_layers, num_point_per_layer = result_np.shape[:3]
            coords = [
                np.arange(num_data),
                np.arange(num_layers),
                np.arange(num_point_per_layer),
                ["s1", "s2"],
            ]
            dims = ["index", "layer", "point", "component"]
            result = xr.DataArray(result_np, coords=coords, dims=dims)
        elif len(result_np.shape) == 3:
            num_data, num_z = result_np.shape[:2]
            coords = [np.arange(num_data), np.arange(num_z), ["s1", "s2"]]
            dims = ["index", "point", "component"]
            result = xr.DataArray(result_np, coords=coords, dims=dims)
        else:
            raise NotImplementedError
        return result

    def eig(self, *args, z=None, **kwargs) -> Tuple[ndarray]:
        """
        Returns values and directions of principal stresses evaluated at given heights
        from the reference surface. Calling the function is exactly the same as for
        `calculate_stresses`, see there for the details. Returns the eigenvalues and the
        eigenvectors as a tuple of NumPy arrays.
        """
        if z is None:
            raise NotImplementedError(
                (
                    "This is only for material stresses at the moment, you must specify"
                    " a distance from the reference surface."
                )
            )

        stresses = self.calculate_stresses(*args, z=z, **kwargs)
        sxx, syy, sxy = (
            stresses.sel(component="SXX").values,
            stresses.sel(component="SYY").values,
            stresses.sel(component="SXY").values,
        )
        shape = sxx.shape + (2, 2)
        stress_matrices = np.zeros(shape, dtype=float)
        stress_matrices[..., 0, 0] = sxx
        stress_matrices[..., 1, 1] = syy
        stress_matrices[..., 0, 1] = sxy
        stress_matrices[..., 1, 0] = sxy
        return eigh(stress_matrices)
