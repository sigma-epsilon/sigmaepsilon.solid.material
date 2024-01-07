from typing import Union, Iterable, Optional, Protocol, runtime_checkable, ClassVar
from numbers import Number

from numpy import ndarray

from .enums import MaterialModelType


@runtime_checkable
class StiffnessLike(Protocol):
    """
    Classes that implement this protocol are suitable for providing
    stiffness definition for structures or structural parts.
    """

    model_type: ClassVar[MaterialModelType]
    number_of_stress_components: ClassVar[int]

    def elastic_stiffness_matrix(
        self, strains: Optional[Union[ndarray, None]] = None
    ) -> ndarray:
        """
        A function that returns the material stiffness matrix as a NumPy array.
        The returned matrix should be positive deifnite.

        In an abstract base it would be wise to implement a way to assert the
        quality of the returned data, namely to check that is represents a valid
        material in the sense that it adheres to a numerical model in alignment with
        the known laws of solid mechanics.
        """
        ...

    def calculate_stresses(self, strains: ndarray) -> ndarray:
        """
        A function that returns stresses for strains as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        ...

    def calculate_strains(self, stresses: ndarray) -> ndarray:
        """
        A function that returns strains for stresses as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        ...


@runtime_checkable
class FailureLike(Protocol):
    """
    Classes that implement this protocol are suitable for providing
    failure definition for materials.
    """

    def utilization(self) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.
        """
        ...


@runtime_checkable
class MaterialLike(Protocol):
    """
    Base class for materials.
    """

    @property
    def stiffness(self) -> StiffnessLike:
        """
        Returns the object representation of the stiffness of the material.
        """
        ...

    @stiffness.setter
    def stiffness(self, value: StiffnessLike) -> None:
        """
        Sets the object representation of the material.
        """
        ...

    @property
    def failure_model(self) -> FailureLike:
        """
        Returns the object representation of the failure model of the material.
        """
        ...

    @failure_model.setter
    def failure_model(self, value: Number) -> None:
        """
        Sets the object representation of the failure model.
        """
        ...


@runtime_checkable
class SectionLike(MaterialLike, Protocol):
    """
    Base class for beam sections.
    """

    ...
