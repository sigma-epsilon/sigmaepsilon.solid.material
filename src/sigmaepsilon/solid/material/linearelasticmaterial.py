from typing import Optional, Union, Iterable, ClassVar
from numbers import Number

from numpy import ndarray
import numpy as np

from sigmaepsilon.math import atleastnd

from .proto import StiffnessLike, FailureLike
from .enums import MaterialModelType

__all__ = ["LinearElasticMaterial"]


class LinearElasticMaterial:
    """
    A class for linear elastic materials. To define one you need at least
    a stiffness provider and optionally an implementation of a failure model.
    """

    model_type: ClassVar[
        Union[MaterialModelType, Iterable[MaterialModelType]]
    ] = MaterialModelType.DEFAULT

    def __init__(
        self,
        stiffness: StiffnessLike,
        failure_model: Optional[Union[FailureLike, None]] = None,
    ):
        self._stiffness = stiffness
        self._failure_model = failure_model

    @property
    def stiffness(self) -> StiffnessLike:
        """
        Returns the object representation of the stiffness of the material.
        """
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: StiffnessLike) -> None:
        """
        Sets the object representation of the material.
        """
        self._stiffness = value

    @property
    def failure_model(self) -> FailureLike:
        """
        Returns the object representation of the failure model of the material.
        """
        return self._failure_model

    @failure_model.setter
    def failure_model(self, value: Number) -> None:
        """
        Sets the object representation of the failure model.
        """
        self._failure_model = value

    def elastic_stiffness_matrix(self, *args, **kwargs) -> ndarray:
        """
        Returns the elastic stiffness matrix of the material model.
        """
        if isinstance(self.stiffness, ndarray):
            return self.stiffness
        return self.stiffness.elastic_stiffness_matrix(*args, **kwargs)

    def utilization(
        self,
        *args,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
        device: str = "cpu",
    ) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.

        The strains or stresses are expected in the order

            s11, s22, s33, s23, s13, s12 = sxx, syy, szz, syz, sxz, sxy

        which is the classical unfolding of the 2nd-order Cauchy stiffness tensor.

        Parameters
        ----------
        strains: numpy.ndarray
            A 1d or 2d array of strain components. If it is a 2d array, it is expected that
            `strains[i]` has the meaning of the i-th selection of strain components, therefore
            the length of the second axis must be 6.

        Returns
        -------
        Number or Iterable[Number]
            A single utilization value as a float or several as a 1d array.
        """
        # strains = self._input_as_dense_bulk_ndarray(strains)
        # stresses = self._input_as_dense_bulk_ndarray(stresses)

        if strains is None:
            assert stresses is not None, "Either strains or stresses must be provided"
            _stresses = self._input_as_dense_bulk_ndarray(stresses)
            strains = self.calculate_strains(stresses=_stresses)
            if isinstance(stresses, tuple):
                strains = [strains[:, i] for i in range(strains.shape[-1])]

        if isinstance(strains, (ndarray, list)):
            strains = atleastnd(strains, 2, front=True)

        if stresses is None:
            stresses = self.calculate_stresses(strains=strains)

        return self.failure_model.utilization(
            *args, strains=strains, stresses=stresses, device=device
        )

    def calculate_stresses(self, *, strains: ndarray) -> ndarray:
        """
        A function that returns stresses for strains as either an 1d or a 2d NumPy array,
        depending on the shape of the input array. Stresses are expected in the order

            xx, yy, zz, yz, xz, xy

        """
        strains = self._input_as_dense_bulk_ndarray(strains)
        return self.stiffness.calculate_stresses(strains)

    def calculate_strains(self, *, stresses: ndarray) -> ndarray:
        """
        A function that returns strains for stresses as either an 1d or a 2d NumPy array,
        depending on the shape of the input array. Strains are expected in the order

            xx, yy, zz, yz, xz, xy

        """
        stresses = self._input_as_dense_bulk_ndarray(stresses)
        return self.stiffness.calculate_strains(stresses)

    def _input_as_dense_bulk_ndarray(self, input: Union[ndarray, tuple, None]):
        if input is None:
            return None
        elif isinstance(input, ndarray):
            result = input
        elif isinstance(input, tuple):
            result = np.stack(input, axis=1)
        return atleastnd(result, 2, front=True)
