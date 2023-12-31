from typing import Union, Optional, Iterable, Tuple
from numbers import Number
import warnings

from numpy import ndarray, ascontiguousarray as ascont
import numpy as np

from sigmaepsilon.math import atleastnd, atleast1d
from sigmaepsilon.math import to_range_1d

from .mindlinshell import MindlinShellLayer, MindlinShellSection
from ..utils.hmh import HMH_S_multi
from ..utils.mindlin import z_to_shear_factors
from ..warnings import SigmaEpsilonMaterialWarning
from ..enums import ModelType

__all__ = ["KirchhoffPlateSection"]


class KirchhoffPlateSection(MindlinShellSection):
    """
    A class for Kirchhoff-Love plates.

    Example
    -------
    >>> from sigmaepsilon.solid.material import KirchhoffPlateSection as Section
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
    (3, 3)

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

    model_type = ModelType.PLATE_KIRCHHOFF_LOVE
    
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

        self._ABDS = ascont(ABDS[3:6, 3:6])
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS
            
    def _postprocess_standard_form(
        self,
        *,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
        shear_forces: Optional[Union[ndarray, None]] = None,
        z: Optional[Union[Number, Iterable[Number], None]] = None,
        rng: Optional[Tuple[Number, Number]] = (-1.0, 1.0),
        squeeze: Optional[bool] = True,
        mode: Optional[str] = "stress",
        layers: Optional[Union[Iterable[MindlinShellLayer], None]] = None,
    ) -> Union[Number, Iterable[Number]]:
        
        if strains is None:
            assert isinstance(stresses, ndarray)
            stresses = atleastnd(stresses, 2, front=True)
            strains = (self.SDBA @ stresses[:, :3].T).T

        strains = atleastnd(strains, 2, front=True)

        if stresses is None:
            assert shear_forces is not None
            stresses = np.zeros((strains.shape[0], 5), dtype=float)
            stresses[:, :3] = (self.ABDS @ strains[:, :3].T).T
            stresses[:, 3:] = shear_forces

        return_stresses = mode == "stress"
        return_eq_stresses = mode == "eq"
        return_utilization = mode == "u"

        assert any([return_stresses, return_eq_stresses, return_utilization])

        # mapping input points
        z = atleast1d(z)
        bounds = self.bounds
        z = to_range_1d(z, source=rng, target=(bounds[0, 0], bounds[-1, -1]))

        # mapping points to layers
        if layers is None:
            layers: Iterable[MindlinShellLayer] = self.find_layers(z, rng)

        num_z = len(layers)
        num_data = strains.shape[0]

        assert all([layer is not None for layer in layers])

        if return_stresses:
            result = np.zeros((num_data, num_z, 5), dtype=float)
        else:
            result = np.zeros((num_data, num_z), dtype=float)

        for iz, layer in enumerate(layers):
            C = layer.material_elastic_stiffness_matrix()
            material_stresses = np.zeros((num_data, 5), dtype=float)

            material_stresses[:, :3] = (
                C[:3, :3] @ (z[iz] * strains[:, :3]).T
            ).T

            sfx, sfy = z_to_shear_factors(z[iz], layer._sfx, layer._sfy)
            material_stresses[:, 3] = sfx * stresses[:, -2]
            material_stresses[:, 4] = sfy * stresses[:, -1]

            if return_stresses:
                result[:, iz, :] = material_stresses
            elif return_eq_stresses:
                result[:, iz] = HMH_S_multi(material_stresses)
            elif return_utilization:
                result[:, iz] = (
                    HMH_S_multi(material_stresses) / layer.material.yield_strength
                )

        return np.squeeze(result) if squeeze else result
