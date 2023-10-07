from typing import Iterable, Optional, Generic, TypeVar, Union
from numbers import Number
from abc import abstractmethod

import numpy as np
from numpy import ndarray

__all__ = ["SurfaceSection"]


class SurfaceLayer:
    """
    Base class for layers.
    """

    __loc__ = [-1.0, 0.0, 1.0]
    __shape__ = (8, 8)

    def __init__(
        self,
        *,
        angle: Optional[Number] = 0.0,
        thickness: Optional[Number] = 1.0,
        material: Optional[Union[ndarray, None]] = None,
    ):
        super().__init__()
        self._angle = angle
        self._thickness = thickness
        self._material = material
        self._tmin = -self._thickness / 2
        self._tmax = self._thickness / 2
        self._zi = [self._loc_to_z(loc) for loc in self.__loc__]

    @property
    def material(self) -> ndarray:
        return self._material

    @material.setter
    def material(self, value: ndarray) -> None:
        if isinstance(value, ndarray):
            self._material = value
        else:
            raise TypeError(f"Expected ndarray, got {type(value)}.")

    @property
    def angle(self) -> Number:
        return self._angle

    @angle.setter
    def angle(self, value: Number) -> None:
        if isinstance(value, Number):
            self._angle = value
        else:
            raise TypeError(f"Expected Number, got {type(value)}.")

    @property
    def thickness(self) -> Number:
        return self._thickness

    @thickness.setter
    def thickness(self, value: Number) -> None:
        if isinstance(value, Number):
            self._thickness = value
        else:
            raise TypeError(f"Expected Number, got {type(value)}.")

    def _loc_to_z(self, loc: Number) -> float:
        """
        Returns height of a local point by linear interpolation.
        Local coordinate is expected between -1 and 1.
        """
        return 0.5 * ((self._tmax + self._tmin) + loc * (self._tmax - self._tmin))

    @abstractmethod
    def elastic_stiffness_matrix(self) -> ndarray:
        raise NotImplementedError


T = TypeVar("T", bound=SurfaceLayer)


class SurfaceSection(Generic[T]):
    layer_class: T = SurfaceLayer

    def __init__(self, layers: Iterable[T], eccentricity: Optional[Number] = 0.0):
        self._layers = layers
        self._eccentricity = eccentricity

    @property
    def layers(self) -> Iterable[T]:
        return self._layers

    @property
    def eccentricity(self) -> Number:
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, value: Number) -> None:
        if isinstance(value, Number):
            self._eccentricity = value
        else:
            raise TypeError(f"Expected Number, got {type(value)}.")

    @classmethod
    def Layer(cls, *args, **kwargs) -> T:
        """
        Returns a Layer compatible with the model.
        """
        return cls.layer_class(*args, **kwargs)

    def _set_layers(self) -> None:
        """
        Sets thickness ranges for the layers.
        """
        ecc = self.eccentricity
        layers = self.layers
        t = sum([layer.thickness for layer in layers])
        layers[0]._tmin = ecc - t / 2
        nLayers = len(layers)

        for i in range(nLayers - 1):
            layers[i]._tmax = layers[i]._tmin + layers[i].thickness
            layers[i + 1]._tmin = layers[i]._tmax
        layers[-1]._tmax = ecc + t / 2

        for layer in layers:
            layer._zi = [layer._loc_to_z(l_) for l_ in layer.__loc__]

    def _elastic_stiffness_matrix(self, out: ndarray) -> None:
        for layer in self._layers:
            out += layer.elastic_stiffness_matrix()

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        Assembles and returns the stiffness matrix.
        """
        self._set_layers()
        self._ABDS = np.zeros(self.layer_class.__shape__)
        self._elastic_stiffness_matrix(self._ABDS)
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS
