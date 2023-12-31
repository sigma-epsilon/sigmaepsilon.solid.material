from typing import Union, Optional, Iterable
from numbers import Number

import numpy as np
from numpy import ndarray

from .meta import BaseMaterialLike
from .elasticitytensor import ElasticityTensor


class LinearElasticMaterial:
    
    def __init__(self, base: BaseMaterialLike, yield_stress: Number=np.Infinity, **params):
        self._base=base
        self._yield_stress = yield_stress
    
    @property
    def base(self) -> BaseMaterialLike:
        return self.base
    
    @base.setter
    def base(self, value: BaseMaterialLike) -> None:
        self.base = value
        
    def elastic_stiffness_matrix(self, *args, **kwargs) -> ndarray:
        return self.base.elastic_stiffness_matrix(*args, **kwargs)
    
    def utilization(
        self,
        stresses: Optional[Union[ndarray, None]] = None,
        strains: Optional[Union[ndarray, None]] = None,
    ) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.
        """
        raise NotImplementedError
    
    def swarm(self, n:int) -> None:
        """
        Sould only ba able to called for single materials (not bulk) and
        then it would clone the seed for a number of 'n' times and go from
        single point to multi point representation. It is important this is also
        where swarming of the frames must happen. 
        """
        ...