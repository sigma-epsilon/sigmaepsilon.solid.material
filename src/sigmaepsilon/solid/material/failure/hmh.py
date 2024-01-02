from typing import Optional, Tuple, Union, ClassVar, Iterable, MutableMapping
from collections import defaultdict

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
from ..enums import MaterialModelType
from ..evaluator import FunctionEvaluator


__all__ = [
    "HuberMisesHenckyFailureCriterion",
    "HuberMisesHenckyFailureCriterion_SP",
    "HuberMisesHenckyFailureCriterion_M",
]


class HuberMisesHenckyFailureCriterion:
    """
    A class to represent the Huber-Mises-Hencky yield condition.

    Instances are callable and can handle arguments in many forms.

    Parameters
    ----------
    yield_strength: float
    """

    model_type: ClassVar[
        Union[MaterialModelType, Iterable[MaterialModelType]]
    ] = MaterialModelType.DEFAULT

    def __init__(
        self,
        yield_strength: Optional[float] = np.Infinity,
        evaluator: Optional[Union[FunctionEvaluator, None]] = None,
    ):
        if isinstance(evaluator, FunctionEvaluator):
            evaluator = defaultdict(lambda: evaluator)

        if evaluator is None:
            evaluator = dict(
                vectorized=FunctionEvaluator(HMH_3d_v, vectorized=True),
                bulk=FunctionEvaluator(HMH_3d_multi, bulk=True),
            )

        self._yield_strength = yield_strength
        self._evaluator = evaluator

    @property
    def yield_strength(self) -> float:
        """
        Returns the yield strength.
        """
        return self._yield_strength

    @property
    def evaluator(self) -> MutableMapping[str, FunctionEvaluator]:
        """
        Returns a dictionary that returns all the possible backend
        implementations, each for different shapes of inputs.
        """
        return self._evaluator

    def _utilization(self, stresses: Union[ndarray, Tuple[ndarray]]) -> ndarray:
        if isinstance(stresses, tuple):
            evaluator = self.evaluator.get("vectorized", None)
            assert isinstance(evaluator, FunctionEvaluator) and evaluator.is_vectorized
            return evaluator(*stresses) / self.yield_strength
        else:
            evaluator = self.evaluator.get("bulk", None)
            if (isinstance(evaluator, FunctionEvaluator) and evaluator.is_bulk):
                return evaluator(stresses) / self.yield_strength
            elif (isinstance(evaluator, FunctionEvaluator) and evaluator.is_vectorized):
                stresses = [stresses[:, i] for i in range(stresses.shape[-1])]
                return evaluator(*stresses) / self.yield_strength
            else:  # pragma: no cover
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

        if isinstance(self.model_type, Iterable):
            num_component_expected = self.model_type[0].number_of_stress_variables
        else:
            num_component_expected = self.model_type.number_of_stress_variables
        
        if not num_component == num_component_expected:  # pragma: no cover
            raise ValueError(
                f"Expected {num_component_expected} stress components, got {num_component}"
            )

        result = self._utilization(stresses)

        return result if not squeeze else np.squeeze(result)

    def __call__(self, *args, **kwargs) -> Union[ndarray, float]:
        return self.utilization(*args, **kwargs)


class HuberMisesHenckyFailureCriterion_SP(HuberMisesHenckyFailureCriterion):
    """
    A class to represent the Huber-Mises-Hencky yield condition
    for plates and shells.

    Parameters
    ----------
    yield_strength: float
    """

    model_type = [
        MaterialModelType.SHELL_UFLYAND_MINDLIN,
        MaterialModelType.SHELL_KIRCHHOFF_LOVE,
    ]

    def __init__(self, *args, evaluator=None, **kwargs):
        if evaluator is None:
            evaluator = dict(
                vectorized=FunctionEvaluator(HMH_S_v, vectorized=True),
                bulk=FunctionEvaluator(HMH_S_multi, bulk=True),
            )
        super().__init__(*args, evaluator=evaluator, **kwargs)


class HuberMisesHenckyFailureCriterion_M(HuberMisesHenckyFailureCriterion):
    """
    A class to represent the Huber-Mises-Hencky yield condition
    for membranes.

    Parameters
    ----------
    yield_strength: float
    """

    model_type = MaterialModelType.MEMBRANE

    def __init__(self, *args, evaluator=None, **kwargs):
        if evaluator is None:
            evaluator = dict(
                vectorized=FunctionEvaluator(HMH_M_v, vectorized=True),
                bulk=FunctionEvaluator(HMH_M_multi, bulk=True),
            )
        super().__init__(*args, evaluator=evaluator, **kwargs)
