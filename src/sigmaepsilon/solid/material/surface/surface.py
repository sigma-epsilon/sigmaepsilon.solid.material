from typing import Iterable, Optional, Generic, TypeVar, Union, Tuple
from numbers import Number
from abc import abstractmethod

import numpy as np
from numpy import ndarray

from ..meta import BaseMaterialLike
from ..elasticitytensor import ElasticityTensor
from ..enums import ModelType

__all__ = ["SurfaceSection"]


class SurfaceLayer:
    """
    Base class for layers.
    """

    __loc__ = [-1.0, 0.0, 1.0]
    __shape__ = (8, 8)

    model_type: ModelType = ModelType.DEFAULT

    def __init__(
        self,
        *,
        angle: Optional[Number] = 0.0,
        thickness: Optional[Number] = 1.0,
        material: Optional[Union[BaseMaterialLike, ndarray, None]] = None,
    ):
        super().__init__()
        self._angle = angle
        self._thickness = thickness
        self._material = material
        self._tmin = -self._thickness / 2
        self._tmax = self._thickness / 2
        self._zi = [self._loc_to_z(loc) for loc in self.__loc__]

    @property
    def material(self) -> Union[BaseMaterialLike, ndarray, None]:
        return self._material

    @material.setter
    def material(self, value: Union[BaseMaterialLike, ndarray, None]) -> None:
        if isinstance(value, (BaseMaterialLike, ndarray)):
            self._material = value
        else:
            if value is not None:
                raise TypeError(f"Expected ndarray, got {type(value)}.")
            self._material = None

    @property
    def material_stiffness_matrix(self) -> ndarray:
        if isinstance(self.material, ElasticityTensor):
            return self.material.contracted_components()
        elif isinstance(self.material, BaseMaterialLike):
            return self.material.elastic_stiffness_matrix()
        elif isinstance(self.material, ndarray):
            return self.material
        raise ValueError("Invaluid material")  # pragma: no cover

    @material_stiffness_matrix.setter
    def material_stiffness_matrix(self, value: ndarray) -> ndarray:
        if isinstance(value, ndarray):
            self.material = value
        else:
            raise TypeError("Invaluid material")  # pragma: no cover

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
    def elastic_stiffness_matrix(self, *_, **__) -> ndarray:
        raise NotImplementedError


T = TypeVar("T", bound=SurfaceLayer)


class SurfaceSection(Generic[T]):
    """
    A generic base class for all kinds of surfaces. Every surface consists of 
    layers. Every layer has a base material.
    """
    layer_class: T = SurfaceLayer
    model_type: ModelType = ModelType.DEFAULT

    def __init__(self, layers: Iterable[T], eccentricity: Optional[Number] = 0.0):
        self._layers = layers
        self._eccentricity = eccentricity
        self._ABDS = None
        self._SDBA = None

    @property
    def SDBA(self) -> ndarray:
        """
        Returns the inverse of the so called 'ABDS' matrix of the shell.
        """
        return self._SDBA

    @property
    def ABDS(self) -> ndarray:
        """
        Returns the so called 'ABDS' matrix of the shell.
        """
        return self._ABDS

    @property
    def layers(self) -> Iterable[T]:
        """
        Returns the layers.
        """
        return self._layers

    @property
    def eccentricity(self) -> Number:
        """
        Returns the eccentricity.
        """
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, value: Number) -> None:
        """
        Sets the eccentricity.
        """
        if isinstance(value, Number):
            self._eccentricity = value
        else:
            raise TypeError(f"Expected Number, got {type(value)}.")

    @property
    def bounds(self) -> ndarray:
        """
        Returns the bounds of the surface as a 2d NumPy array.
        """
        return np.array([layer._zi for layer in self.layers], dtype=float)

    @property
    def angles(self) -> ndarray:
        """
        Returns the angles of the layers as an 1d NumPy array.
        """
        return np.array([layer.angle for layer in self.layers], dtype=float)

    @property
    def thickness(self) -> Number:
        """
        Returns the thickness.
        """
        return sum([l.thickness for l in self.layers])

    def find_layer(
        self, z: float, rng: Optional[Iterable[Number]] = (-1, 1), tol: float = 1e-3
    ) -> T:
        """
        Returns the layer that contains the point
        """
        layers = self.layers
        z = float(z)
        for layer in layers:
            z_min = layer._loc_to_z(-1.0)
            z_max = layer._loc_to_z(1.0)
            if z > (z_min - tol) and z <= (z_max + tol):
                return layer
        return None

    def find_layers(
        self, z: Iterable[float], rng: Optional[Iterable[Number]] = (-1, 1)
    ) -> Iterable[T]:
        """
        Returns the hosting layer for several points along the thickness.
        """
        return [self.find_layer(zi, rng) for zi in z]

    @classmethod
    def Layer(cls, *args, **kwargs) -> T:
        """
        Returns a Layer compatible with the model.
        """
        return cls.layer_class(*args, **kwargs)

    @staticmethod
    def Material(**kwargs) -> ndarray:
        """
        Ought to return the material represented as a 2d NumPy array.
        """
        raise NotImplementedError

    def _set_layers(self) -> None:
        """
        Sets thickness ranges for the layers.
        """
        ecc = self.eccentricity
        layers: Iterable[T] = self.layers
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

    def elastic_stiffness_matrix(self, *_, **__) -> ndarray:
        """
        Assembles and returns the stiffness matrix.
        """
        self._set_layers()
        self._ABDS = np.zeros(self.layer_class.__shape__)
        self._elastic_stiffness_matrix(self._ABDS)
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS

    def utilization(self, *args, **kwargs) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.

        The implementation should be able to cover the case if the input 'strains' is a 2d array.
        In that case, the strain values are expected to run along the last axis, hence the i-th
        item would be accessed as `strains[i]` and it would return a tuple of numbers, one for
        every strain component involved in the formulation of the material law.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d array of strains such that the strains run along the last axis.
            The shape of this array determines the shape of the output in a straightforward
            manner.
        strains: numpy.ndarray, Optional
            1d or 2d array of strains such that the strains run along the last axis.
            The shape of this array determines the shape of the output in a straightforward
            manner.
        z: float or None, Optional
            The signed normal distance from the reference surface of the body.
            If `None`, values for all points are returned, grouped by layers (see later).
            Use it in combination with the 'rng' parameter.
        rng: Tuple[Number, Number], Optional
            An interval that puts the value of 'z' to perspective. Otherwise specified, the
            value for the parameter 'z' is expected in the range [-1, 1].

        Note
        ----
        The returned result treats layers as iterables even if the case of one single layer.
        This shows in the shapes of output arrays and you will quickly find the logic behind it
        with minimal experimentation.
        """
        raise NotImplementedError
        
    def calculate_stresses(self, *args, **kwargs) -> ndarray:
        """
        Calculates material stresses for input internal forces or strains
        and returns it as a NumPy array.

        Either strains or stresses must be provided.

        If the points of evaluation are not explivitly specified with the parameter 'z',
        results are calculated at a default number of points per every layer.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None.
        stresses: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None
        z: Iterable[Number], Optional
            Points of evaluation. Default is None.
        rng: Iterable[Number], Optional
            The range in which 'z' is to be understood. Default is (-1, 1).
        squeeze: bool, Optional
            Whether to squeeze the reusulting array or not. Default is `True`.
        ppl: int, Optional
            Point per layer. Default is None.
        """
        raise NotImplementedError
    
    def calculate_equivalent_stress(self, *args, **kwargs) -> ndarray:
        """
        Calculates equivalent material stresses for input internal forces or strains 
        according to the built-in failure criteria and returns it as a NumPy array.

        Either strains or stresses must be provided.

        If the points of evaluation are not explivitly specified with the parameter 'z',
        results are calculated at a default number of points per every layer.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None.
        stresses: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None
        z: Iterable[Number], Optional
            Points of evaluation. Default is None.
        rng: Iterable[Number], Optional
            The range in which 'z' is to be understood. Default is (-1, 1).
        squeeze: bool, Optional
            Whether to squeeze the reusulting array or not. Default is `True`.
        ppl: int, Optional
            Point per layer. Default is None.
        """
        raise NotImplementedError
