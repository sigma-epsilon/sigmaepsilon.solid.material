from typing import Optional, Union, Iterable, ClassVar
from numbers import Number

from numpy import ndarray

from .proto import StiffnessLike, FailureLike
from .enums import MaterialModelType


class LinearElasticMaterial:
    """
    A class for linear elastic materials.
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

    def utilization(self, *args, **kwargs) -> ndarray:
        """
        Returns the elastic stiffness matrix of the material model.
        """
        return self.failure_model.utilization(*args, **kwargs)
