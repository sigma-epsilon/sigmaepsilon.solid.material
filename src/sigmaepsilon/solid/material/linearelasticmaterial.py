from numbers import Number

import numpy as np
from numpy import ndarray

from .proto import StiffnessLike


class LinearElasticMaterial:
    """
    A class for linear elastic materials.
    """

    def __init__(self, stiffness: StiffnessLike, yield_strength: Number = np.Infinity):
        self._stiffness = stiffness
        self._yield_strength = yield_strength

    @property
    def stiffness(self) -> StiffnessLike:
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: StiffnessLike) -> None:
        self._stiffness = value

    @property
    def yield_strength(self) -> Number:
        return self._yield_strength

    @yield_strength.setter
    def yield_strength(self, value: Number) -> None:
        self._yield_strength = value

    def elastic_stiffness_matrix(self, *args, **kwargs) -> ndarray:
        return self.stiffness.elastic_stiffness_matrix(*args, **kwargs)
