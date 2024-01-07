from typing import (
    Optional,
    Tuple,
    Union,
    ClassVar,
    Iterable,
    MutableMapping,
    List,
    Callable,
)

import numpy as np
from numpy import ndarray

from sigmaepsilon.core.kwargtools import allinkwargs, getallfromkwargs
from sigmaepsilon.math import atleast2d

from ..utils.hoffman import (
    Hoffman_failure_criterion_principal_form,
    Hoffman_failure_criterion_principal_form_PS,
    Hoffman_failure_criterion_principal_form_M,
)
from ..enums import MaterialModelType
from ..evaluator import FunctionEvaluator
from .abstract import AbstractFailureCriterion

__all__ = [
    "HoffmanFailureCriterion",
    "HoffmanFailureCriterion_SP",
    "HoffmanFailureCriterion_M",
]


class HoffmanFailureCriterion(AbstractFailureCriterion):
    """
    A class to represent the Hoffman failure criterion.

    Instances are callable and can handle arguments in many shapes.
    """

    model_type: ClassVar[
        Union[MaterialModelType, Iterable[MaterialModelType]]
    ] = MaterialModelType.DEFAULT

    principal_model_params: ClassVar[List[int]] = [
        "Xc",
        "Xt",
        "Yc",
        "Yt",
        "Zc",
        "Zt",
        "S23",
        "S13",
        "S12",
    ]

    failure_evaluator: ClassVar[Callable] = dict(
        vectorized=FunctionEvaluator(
            Hoffman_failure_criterion_principal_form, vectorized=True
        )
    )

    number_of_stress_arguments: ClassVar[int] = 6

    @property
    def number_of_strength_parameters(self) -> int:
        return len(self.__class__.principal_model_params)

    def __init__(
        self,
        *,
        params: Optional[Union[Iterable[float], None]] = None,
        evaluator: Optional[Union[FunctionEvaluator, None]] = None,
        **kwargs,
    ):
        n_params = self.number_of_strength_parameters

        if params is None:
            if (len(kwargs) == n_params) and allinkwargs(
                self.__class__.principal_model_params, **kwargs
            ):
                params = tuple(
                    getallfromkwargs(self.__class__.principal_model_params, **kwargs)
                )
            else:
                params = tuple(
                    getallfromkwargs(
                        self.__class__.principal_model_params,
                        default=np.Infinity,
                        **kwargs,
                    )
                )

        if params is not None:
            self._params = tuple(params)

        if evaluator is None:
            evaluator = self.__class__.failure_evaluator

        self._evaluator = evaluator

    @property
    def params(self) -> Iterable[float]:
        """
        Returns the strength parameters.
        """
        return self._params

    @params.setter
    def params(self, value) -> None:
        """
        Sets the strength parameters.
        """
        self._params = value

    @property
    def evaluator(self) -> MutableMapping[str, FunctionEvaluator]:
        """
        Returns a dictionary that returns all the possible backend
        implementations, each for different shapes of inputs.
        """
        return self._evaluator

    def _utilization(
        self,
        stresses: Union[ndarray, Tuple[ndarray]],
        params: Optional[Union[Iterable[float], None]] = None,
    ) -> ndarray:
        if params is None:
            params = self.params

        if isinstance(stresses, tuple):
            num_s = self.__class__.number_of_stress_arguments
            if not len(stresses) == num_s:
                raise ValueError(
                    f"Expected {num_s} stress components, got {len(stresses)}"
                )
            evaluator = self.evaluator.get("vectorized", None)
            assert isinstance(evaluator, FunctionEvaluator) and evaluator.is_vectorized
            return evaluator(*(stresses + params))
        else:
            num_s = self.__class__.number_of_stress_arguments
            if not stresses.shape[-1] == num_s:
                raise ValueError(
                    f"Expected {num_s} stress components, got {stresses.shape[-1]}"
                )
            evaluator = self.evaluator.get("vectorized", None)
            if isinstance(evaluator, FunctionEvaluator) and evaluator.is_vectorized:
                stresses = tuple([stresses[:, i] for i in range(stresses.shape[-1])])
                return evaluator(*(stresses + params))
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
            model_type: MaterialModelType = self.model_type[0]
            num_component_expected = model_type.number_of_material_stress_variables
        else:
            num_component_expected = self.model_type.number_of_material_stress_variables

        if not num_component == num_component_expected:  # pragma: no cover
            raise ValueError(
                f"Expected {num_component_expected} stress components, got {num_component}"
            )

        result = self._utilization(stresses)

        return result if not squeeze else np.squeeze(result)

    def __call__(self, *args, **kwargs) -> Union[ndarray, float]:
        return self.utilization(*args, **kwargs)


class HoffmanFailureCriterion_SP(HoffmanFailureCriterion):
    """
    A class to represent the Hoffmann yield condition
    for plates and shells.

    Parameters
    ----------
    yield_strength: float
    """

    model_type = [
        MaterialModelType.SHELL_UFLYAND_MINDLIN,
        MaterialModelType.SHELL_KIRCHHOFF_LOVE,
    ]

    failure_evaluator: ClassVar[Callable] = dict(
        vectorized=FunctionEvaluator(
            Hoffman_failure_criterion_principal_form_PS, vectorized=True
        )
    )

    number_of_stress_arguments: ClassVar[int] = 5


class HoffmanFailureCriterion_M(HoffmanFailureCriterion):
    """
    A class to represent the Hoffman yield condition
    for membranes.
    """

    model_type = MaterialModelType.MEMBRANE

    principal_model_params: ClassVar[List[int]] = [
        "Xc",
        "Xt",
        "Yc",
        "Yt",
        "Zc",
        "Zt",
        "S12",
    ]

    failure_evaluator: ClassVar[Callable] = dict(
        vectorized=FunctionEvaluator(
            Hoffman_failure_criterion_principal_form_M, vectorized=True
        )
    )

    number_of_stress_arguments: ClassVar[int] = 3
