from typing import Optional, Union, ClassVar, Iterable, Callable, MutableMapping
from abc import abstractproperty, abstractmethod

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm as BGA

from ..enums import MaterialModelType
from ..evaluator import FunctionEvaluator


class AbstractFailureCriterion:
    """
    Base class for all phenomenological failure models in the library.
    """

    model_type: ClassVar[Union[MaterialModelType, Iterable[MaterialModelType]]]
    number_of_stress_arguments: ClassVar[int]
    failure_evaluator: ClassVar[Optional[Union[Callable, None]]] = None

    @abstractproperty
    def number_of_strength_parameters(self) -> int: ...

    @abstractproperty
    def params(self) -> Iterable[float]: ...

    @property
    def evaluator(self) -> MutableMapping[str, FunctionEvaluator]:
        """
        Returns a dictionary that returns all the possible backend
        implementations, each for different shapes of inputs.
        """
        ...

    @abstractmethod
    def utilization(self, *args, **kwargs) -> Union[ndarray, float]: ...

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
                failure_obj = self.__class__(params=parameters)
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
            if num_params > 1:
                assert len(x0) == num_params
            res = minimize(
                objective, x0, method="Nelder-Mead", tol=tol, **solver_params
            )
            params = res.x
        else:
            raise NotImplementedError(f"Unknown method '{method}'")

        return tuple(params)
