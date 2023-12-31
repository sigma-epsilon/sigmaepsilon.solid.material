from typing import (
    Union,
    Iterable,
    Optional,
    Protocol,
    runtime_checkable,
)
from numbers import Number

from numpy import ndarray


@runtime_checkable
class BaseMaterialLike(Protocol):
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

    @property
    def yield_strength(self) -> Number:
        """
        The maximum stress the material can sustain without experiencing 
        relevant losses in performance. This means that there must be a procedure
        assigned with each material object dedicated for bringing the stress state
        to an invariant value. In other words, each instance needs to have an inplementation
        to identify if the state of stress is in the elastic or plastic region and that
        this expression must be unique.

        For an abstract base class, a numrically infinite value should work.
        In this case, the recommended way is to use `np.Infinity` from NumPy.
        """
        ...

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
        raise NotImplementedError

    def utilization(self, strains: ndarray) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.
        """
        raise NotImplementedError
    
    def calculate_stresses(self, strains: ndarray) -> ndarray:
        """
        A function that returns stresses for strains as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        raise NotImplementedError


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

    def swarm(self, n: int) -> None:
        """
        Sould only ba able to called for single materials (not bulk) and
        then it would clone the seed for a number of 'n' times and go from
        single point to multi point representation. It is important this is also
        where swarming of the frames must happen.
        """
        ...


@runtime_checkable
class SectionLike(MaterialLike, Protocol):
    """
    Base class for beam sections.
    """

    ...
