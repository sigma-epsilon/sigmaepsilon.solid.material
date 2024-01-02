from typing import Optional, Union, Iterable, ClassVar, Tuple
from numbers import Number

from numpy import ndarray
import numpy as np

from .proto import StiffnessLike, FailureLike
from .enums import MaterialModelType


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
        return self.stiffness.elastic_stiffness_matrix(*args, **kwargs)

    def utilization(
        self,
        *args,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
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
        return (
            self.calculate_equivalent_stress(*args, strains=strains, stresses=stresses)
            / self.yield_strength
        )

    def calculate_equivalent_stress(
        self,
        *args,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
    ) -> ndarray:
        """
        A function that calculates the equivalent stress.

        The strains are expected in the order

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
            A single value as a float or several as a 1d array.
        """
        if strains is None and stresses is None:
            if len(args) == 1:
                if isinstance(args[0], ndarray):
                    return self.calculate_equivalent_stress(strains=args[0])
            elif len(args) == 6:
                return self.calculate_equivalent_stress(strains=args)

        if isinstance(strains, Tuple):
            if len(strains) == 6:
                stresses = self.contracted_components() @ np.stack(strains, axis=0)
                s11, s22, s33, s23, s13, s12 = stresses
                #return HMH_3d_v(s11, s22, s33, s23, s13, s12)

            raise ValueError(
                (
                    "Invalid shape. If 'strains' is a tuple, it must be a 6-tuple "
                    "of 1d NumPy arrays of the same length."
                )
            )

        if not strains.shape[-1] == 6:
            raise ValueError("Invalid number of strain components.")

        if len(strains.shape) == 1:
            stresses = self.elastic_stiffness_matrix() @ strains
            #result = HMH_3d(stresses)
        else:
            stresses = (self.elastic_stiffness_matrix() @ strains.T).T
            #result = HMH_3d_multi(stresses)

        #return result

    def calculate_stresses(self, strains: ndarray) -> ndarray:
        """
        A function that returns stresses for strains as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        if not strains.shape[-1] == 6:
            raise ValueError("Invalid number of strain components.")

        stresses = None

        if len(strains.shape) == 1:
            stresses = self.elastic_stiffness_matrix() @ strains
        else:
            stresses = (self.elastic_stiffness_matrix() @ strains.T).T

        return stresses

    def calculate_strains(self, stresses: ndarray) -> ndarray:
        """
        A function that returns strains for stresses as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        if not stresses.shape[-1] == 6:
            raise ValueError("Invalid number of strain components.")

        strains = None

        if len(stresses.shape) == 1:
            strains = np.linalg.inv(self.elastic_stiffness_matrix()) @ stresses
        else:
            strains = (np.linalg.inv(self.elastic_stiffness_matrix()) @ stresses.T).T

        return strains