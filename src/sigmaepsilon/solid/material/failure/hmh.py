from typing import Optional, Tuple, Union, ClassVar, Iterable, MutableMapping
from collections import defaultdict
from numbers import Number

from xarray import DataArray
import numpy as np
from numpy import ndarray, ascontiguousarray as ascont

from sigmaepsilon.math import atleast2d

from ..utils.hmh import (
    HMH_3d_v,
    HMH_3d_multi,
    HMH_3d_v_cuda,
    HMH_3d_guv_cuda,
    HMH_3d_v_cuda_cp,
    HMH_3d_v_numba_cuda_kernel,
    HMH_S_multi,
    HMH_S_v,
    HMH_M_multi,
    HMH_M_v,
    divide_array,
)
from ..enums import MaterialModelType
from ..evaluator import FunctionEvaluator
from .abstract import AbstractFailureCriterion

__all__ = [
    "HuberMisesHenckyFailureCriterion",
    "HuberMisesHenckyFailureCriterion_SP",
    "HuberMisesHenckyFailureCriterion_M",
]


class HuberMisesHenckyFailureCriterion(AbstractFailureCriterion):
    """
    A class to represent the Huber-Mises-Hencky yield condition.

    Instances are callable and can handle arguments in many forms.

    Parameters
    ----------
    yield_strength: float
    """

    model_type = MaterialModelType.DEFAULT
    number_of_stress_arguments: ClassVar[int] = 1
    failure_evaluator = None

    def __init__(
        self,
        yield_strength: Optional[float] = np.Infinity,
        evaluator: Optional[Union[FunctionEvaluator, None]] = None,
    ):
        if isinstance(evaluator, FunctionEvaluator):
            evaluator = defaultdict(lambda: evaluator)

        if evaluator is None:
            evaluator = {
                "vectorized": FunctionEvaluator(HMH_3d_v, vectorized=True),
                "bulk": FunctionEvaluator(HMH_3d_multi, bulk=True),
                "cuda": FunctionEvaluator(HMH_3d_v_cuda, vectorized=True, cuda=True),
                "cuda-guv": FunctionEvaluator(
                    HMH_3d_guv_cuda, vectorized=True, cuda=True
                ),
                "cuda-kernel": FunctionEvaluator(
                    HMH_3d_v_numba_cuda_kernel, vectorized=True, cuda=True
                ),
                "cuda-cp": FunctionEvaluator(
                    HMH_3d_v_cuda_cp, vectorized=True, cuda=True
                ),
            }

        self._yield_strength = yield_strength
        self._evaluator = evaluator

    @property
    def number_of_strength_parameters(self) -> int:
        return 1

    @property
    def params(self) -> Iterable[float]:
        """
        Returns the strength parameters.
        """
        return [self.yield_strength]

    @params.setter
    def params(self, value) -> None:
        """
        Sets the strength parameters.
        """
        if isinstance(value, Iterable):
            if not len(value) == 1:
                raise ValueError("Only one parameter is expected")
            self._yield_strength = value[0]
        elif isinstance(value, Number):
            self._yield_strength = value
        else:
            raise TypeError("Invalid type.")

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

    def _utilization(
        self, stresses: Union[ndarray, Tuple[ndarray]], device: str = "cpu"
    ) -> ndarray:
        if isinstance(stresses, tuple):
            if device == "cpu":
                evaluator = self.evaluator.get("vectorized", None)
                assert (
                    isinstance(evaluator, FunctionEvaluator) and evaluator.is_vectorized
                )
                return evaluator(*stresses) / self.yield_strength
            elif device == "cuda":
                evaluator = self.evaluator.get("cuda", None)
                assert (
                    isinstance(evaluator, FunctionEvaluator)
                    and evaluator.is_vectorized
                    and evaluator.is_cuda
                )
                return evaluator(*stresses) / self.yield_strength
            elif device == "cuda-guv":
                evaluator = self.evaluator.get("cuda-guv", None)
                assert (
                    isinstance(evaluator, FunctionEvaluator)
                    and evaluator.is_vectorized
                    and evaluator.is_cuda
                )
                return evaluator(*stresses) / self.yield_strength
            elif device == "cuda-kernel":
                evaluator = self.evaluator.get("cuda-kernel", None)
                assert (
                    isinstance(evaluator, FunctionEvaluator)
                    and evaluator.is_vectorized
                    and evaluator.is_cuda
                )
                output = evaluator(*stresses)
                divide_array(output, self.yield_strength)
                return output
            elif device == "cuda-cp":
                evaluator = self.evaluator.get("cuda-cp", None)
                assert (
                    isinstance(evaluator, FunctionEvaluator)
                    and evaluator.is_vectorized
                    and evaluator.is_cuda
                )
                return evaluator(*stresses) / self.yield_strength
            else:
                raise NotImplementedError(
                    f"This calculation is not implemented for the specified device '{device}'"
                )
        else:
            if device == "cpu":
                evaluator = self.evaluator.get("bulk", None)
                if isinstance(evaluator, FunctionEvaluator) and evaluator.is_bulk:
                    return evaluator(stresses) / self.yield_strength
                elif (
                    isinstance(evaluator, FunctionEvaluator) and evaluator.is_vectorized
                ):
                    stresses = [stresses[:, i] for i in range(stresses.shape[-1])]
                    return evaluator(*stresses) / self.yield_strength
                else:  # pragma: no cover
                    raise NotImplementedError
            elif device == "cuda":
                evaluator = self.evaluator.get("cuda", None)
                assert (
                    isinstance(evaluator, FunctionEvaluator)
                    and evaluator.is_vectorized
                    and evaluator.is_cuda
                )
                stresses = [ascont(stresses[:, i]) for i in range(stresses.shape[-1])]
                return evaluator(*stresses) / self.yield_strength
            elif device == "cuda-cp":
                evaluator = self.evaluator.get("cuda-cp", None)
                assert (
                    isinstance(evaluator, FunctionEvaluator)
                    and evaluator.is_vectorized
                    and evaluator.is_cuda
                )
                stresses = [ascont(stresses[:, i]) for i in range(stresses.shape[-1])]
                return evaluator(*stresses) / self.yield_strength
            else:
                raise NotImplementedError(
                    f"This calculation is not implemented for the specified device '{device}'"
                )

    def utilization(
        self,
        *,
        stresses: Optional[Union[ndarray, Tuple[ndarray], None]] = None,
        squeeze: bool = True,
        device: str = "cpu",
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
        elif isinstance(stresses, DataArray):
            stresses = atleast2d(stresses.values)
            num_component = stresses.shape[-1]
        else:
            stress_type = type(stresses)
            raise TypeError(
                f"Expected a NumPy arrar or a tuple of them, got {stress_type}"
            )

        if isinstance(self.model_type, Iterable):
            num_component_expected = self.model_type[
                0
            ].number_of_material_stress_variables
        else:
            num_component_expected = self.model_type.number_of_material_stress_variables

        if not num_component == num_component_expected:  # pragma: no cover
            raise ValueError(
                f"Expected {num_component_expected} stress components, got {num_component}"
            )

        result = self._utilization(stresses, device)

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
