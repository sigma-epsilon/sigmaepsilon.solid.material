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
from scipy.optimize import minimize

from sigmaepsilon.core.kwargtools import allinkwargs, getallfromkwargs
from sigmaepsilon.math import atleast2d
from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm as BGA

from ..utils.hoffman import (
    Hoffman_failure_criterion_principal_form,
    Hoffman_failure_criterion_principal_form_PS,
    Hoffman_failure_criterion_principal_form_M,
)
from ..enums import MaterialModelType
from ..evaluator import FunctionEvaluator


__all__ = [
    "HoffmanFailureCriterion",
    "HoffmanFailureCriterion_SP",
    "HoffmanFailureCriterion_M",
]


class HoffmanFailureCriterion:
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

    def fit(
        self,
        inputs: ndarray,
        outputs: ndarray,
        solver_params: Optional[Union[dict, None]] = None,
        tol: Optional[float] = 1e-3,
        penalty: Optional[float] = 1e12,
        n_iter: Optional[Union[int, None]] = None,
        ranges: Optional[Union[Iterable[Iterable], None]] = None,
        method: Optional[str] = "auto",
        x0: Optional[Union[Iterable, None]] = None,
    ) -> Iterable[float]:
        """
        Calculates the parameters of the criterion by fitting them to
        observations. Returns the best fit as a tuple.

        The fitting of the parameters is carried out by minimizing an error
        measure using a binary genetic algorithm (BGA) or the Nelder-Mead method.

        Parameters
        ----------
        inputs: numpy.ndarray
            A 2d array of stresses. Each row must have 6 components in the order
                s11, s22, s33, s23, s13, s12
        outputs: numpy.ndarray
            The observations as an 1d NumPy array.
        solver_params: dict, Optional
            Parameters passed on to the specified solver. Only available is the
            method is specified.
        penalty: float, Optional
            Penalty parameter to apply for constraint violations. Default is 1e12.
        n_iter: int, Optional
            Number of iterations. Default is None.
        ranges: Iterable[Iterable], Optional
            Ranges for the unknowns as an iterable of iterables.
            Only for the genetic algorithm. Default is [-10, 10] for each variable.
        x0: Iterable, Optional
            1d iterable for the initial solution, only for the Nelder-Mead method. The
            variables must come in the order
                Xc, Xt, Yc, Yt, Zc, Zt, S23, S13, S12
        method: str, Optional
            Available options:
                * BGA : Binary Genetic Algorithm
                * Nelder-Mead : the Nelder-Mead method (aka. nonlinear simplex method)
        """

        if method == "auto":
            assert solver_params is None
            method = "bga" if x0 is None else "Nelder-Mead"

        if solver_params is None:
            solver_params = dict()

        def objective(parameters):
            try:
                failure_obj = HoffmanFailureCriterion(params=parameters)
                prediction = failure_obj.utilization(stresses=inputs)
                errors = np.sqrt((prediction - outputs) ** 2)
                total_error = np.sum(errors)

                if np.sum(np.abs(parameters)) < tol:
                    total_error += penalty

                return total_error
            except:
                return penalty

        num_params = self.number_of_strength_parameters

        if not isinstance(method, str):
            method_type = type(method)
            raise TypeError(f"Parameter 'method' must be a string, got {method_type}")

        if method.lower() == "bga":
            ranges = (
                [[-10, 10] for _ in range(num_params)] if ranges is None else ranges
            )
            # FIXME: Check after the minimization if the solution touches the ranges
            # or not. If it does, the minimization should be rerun with modified ranges.
            # NOTE: This could be a general extension for the genetic algorithm.
            _solver_params = {
                "length": 8,
                "nPop": 100,
            }
            _solver_params.update(solver_params)
            solver = BGA(objective, ranges, **_solver_params)
            if isinstance(n_iter, int):
                _ = [solver.evolve(1) for _ in range(n_iter)]
            else:
                solver.solve()
            params = solver.best_phenotype()
        elif method == "Nelder-Mead":
            assert x0 is not None
            assert len(x0) == num_params
            res = minimize(
                objective, x0, method="Nelder-Mead", tol=tol, **solver_params
            )
            params = res.x
        else:
            raise NotImplementedError(f"Unknown method '{method}'")

        return tuple(params)

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
