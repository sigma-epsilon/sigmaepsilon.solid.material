from typing import Union, Iterable, Optional, Protocol, runtime_checkable, ClassVar
from numbers import Number

from numpy import ndarray

from .enums import ModelType


@runtime_checkable
class StiffnessLike(Protocol):
    """
    Classes that implement this protocol are suitable for providing
    stiffness definition for structures or structural parts.
    """

    model_type: ClassVar[ModelType]

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


@runtime_checkable
class BaseMaterialLike(StiffnessLike, Protocol):
    """
    Base class for materials.

    Note
    ----
    If the instance stores
    data for multiple entities, their yield strength is considered to
    be the same. Therefore, logical units of a model should be construced
    in a way that uses big blocks of cells of the same material. The fewer
    the blocks and bigger the chunks, the better the efficiency of data structure.
    """

    def utilization(self) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.
        """
        ...

    def calculate_stresses(self) -> ndarray:
        """
        A function that returns stresses for strains as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        ...
    
    def calculate_equivalent_stress(self) -> ndarray:
        """
        A function that returns stresses for strains as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        ...


@runtime_checkable
class MaterialLike(Protocol):
    """
    Base class for materials.
    """

    def elastic_stiffness_matrix(
        self, strains: Optional[Union[ndarray, None]] = None
    ) -> ndarray:
        """
        A function that returns the material stiffness matrix as a NumPy array.
        """
        raise NotImplementedError

    def utilization(
        self,
        strains: Optional[Union[ndarray, None]] = None,
    ) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.
        """
        raise NotImplementedError


@runtime_checkable
class SectionLike(MaterialLike, Protocol):
    """
    Base class for beam sections.
    """

    ...
