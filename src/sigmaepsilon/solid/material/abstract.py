from typing import (
    Protocol,
    runtime_checkable,
    Iterable
)
from numbers import Number

from numpy import ndarray

@runtime_checkable
class MaterialLike(Protocol):
    """
    Base class for materials.
    """

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        A function that returns the material stiffness matrix as a NumPy array.
        """
        raise NotImplementedError
    

    def yields(self) -> bool:
        """
        A function that returns True if the material has reached its maximum capacity.
        A probable argument to accept here are stresses.
        """
        raise NotImplementedError
    
    def utilization(self) -> Number:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.
        """
        ...


@runtime_checkable
class SectionLike(MaterialLike, Protocol):
    """
    Base class for beam sections.
    """
    ...