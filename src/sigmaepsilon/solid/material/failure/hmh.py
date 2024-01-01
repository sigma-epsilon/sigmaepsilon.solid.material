from typing import Optional, Tuple, Union, Iterable

import numpy as np
from numpy import ndarray

from sigmaepsilon.math import atleast2d
from ..utils.hmh import (
    HMH_3d_v,
    HMH_3d_multi,
    HMH_S_multi,
    HMH_S_v,
    HMH_M_multi,
    HMH_M_v,
)


class HuberMisesHenckyFailureTheory:
    """
    A class to represent the Huber-Mises-Hencky yield condition.

    Parameters
    ----------
    yield_strength: float
    """

    def __init__(self, yield_strength: float):
        self._yield_strength = yield_strength

    @property
    def yield_strength(self) -> float:
        """
        Returns the yield strength.
        """
        return self._yield_strength

    def _utilization_3d(self, stresses: Union[ndarray, Tuple[ndarray]]) -> ndarray:
        if isinstance(stresses, ndarray):
            return HMH_3d_multi(stresses) / self.yield_strength
        elif isinstance(stresses, tuple):
            return HMH_3d_v(*stresses) / self.yield_strength
        raise NotImplementedError

    def _utilization_mindlin(self, stresses: Union[ndarray, Tuple[ndarray]]) -> ndarray:
        if isinstance(stresses, ndarray):
            return HMH_S_multi(stresses) / self.yield_strength
        elif isinstance(stresses, tuple):
            return HMH_S_v(*stresses) / self.yield_strength
        raise NotImplementedError

    def _utilization_membrane(
        self, stresses: Union[ndarray, Tuple[ndarray]]
    ) -> ndarray:
        if isinstance(stresses, ndarray):
            return HMH_M_multi(stresses) / self.yield_strength
        elif isinstance(stresses, tuple):
            return HMH_M_v(*stresses) / self.yield_strength
        raise NotImplementedError

    def utilization(
        self,
        *,
        stresses: Optional[Union[ndarray, Tuple[ndarray], None]] = None,
        squeeze: bool = True,
        **__,
    ) -> Union[ndarray, float]:
        """
        Calculates utilization from one ore more set of strains.

        Parameters
        ----------
        stresses: numpy.ndarray
            1d or 2d array of values representing Cauchy stresses calculated
            at one or more points.
        """
        if isinstance(stresses, ndarray):
            stresses = atleast2d(stresses)
            num_component = stresses.shape[-1]
        elif isinstance(stresses, tuple):
            num_component = len(stresses)
        else:
            raise TypeError(
                f"Expected a NumPy arrar or a tuple of them, got {type(stresses)}"
            )

        result = None
        if num_component == 6:
            result = self._utilization_3d(stresses)
        elif num_component == 5:
            result = self._utilization_mindlin(stresses)
        elif num_component == 3:
            result = self._utilization_membrane(stresses)
        else:
            raise ValueError(
                "The input should either have 6, 5 or 3 stress components."
            )

        return result if not squeeze else np.squeeze(result)
