from typing import Union, Optional, Iterable
from numbers import Number

from numpy import ndarray, ascontiguousarray as ascont
import numpy as np

from sigmaepsilon.math import atleastnd

from .mindlinshell import MindlinShellLayer, MindlinShellSection
from ..enums import MaterialModelType

__all__ = ["KirchhoffShellSection"]


class KirchhoffShellLayer(MindlinShellLayer):
    """
    A class for layers of a Kirchhoff-Love shells.
    """
    ...


class KirchhoffShellSection(MindlinShellSection):
    """
    A class for Kirchhoff-Love shells.

    Example
    -------
    >>> from sigmaepsilon.solid.material import KirchhoffShellSection as Section
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
    (6, 6)

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

    model_type = MaterialModelType.SHELL_KIRCHHOFF_LOVE
    layer_class = KirchhoffShellLayer

    def elastic_stiffness_matrix(self, tol: Number = 1e-8) -> ndarray:
        """
        Assembles and returns the stiffness matrix.
        """
        self._set_layers()
        ABDS = np.zeros(self.layer_class.__shape__)
        self._elastic_stiffness_matrix(ABDS)
        self._ABDS = ascont(ABDS[:6, :6])
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS

    def _postprocess_standard_form(
        self,
        *,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
        shear_forces: Optional[Union[ndarray, None]] = None,
        **kwargs
    ) -> Union[Number, Iterable[Number]]:

        if strains is None:
            assert isinstance(stresses, ndarray)
            stresses = atleastnd(stresses, 2, front=True)
            strains = (self.SDBA @ stresses[:, :6].T).T

        strains = atleastnd(strains, 2, front=True)

        if stresses is None:
            assert shear_forces is not None
            stresses = np.zeros((strains.shape[0], 8), dtype=float)
            stresses[:, :6] = (self.ABDS @ strains[:, :6].T).T
            stresses[:, 6:] = shear_forces

        return super()._postprocess_standard_form(strains=strains, stresses=stresses, **kwargs)
