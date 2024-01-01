from typing import Callable, Optional


class FunctionEvaluator(Callable):
    """
    A class to represent different evaluation schemes as attributes of
    a callable. Technickly, an instance of `FunctionEvaluator` wraps
    a callable and attaches attributes to it to explicitly suggest
    behaviour.
    """

    def __init__(
        self,
        fnc: Callable,
        vectorized: Optional[bool] = False,
        bulk: Optional[bool] = False,
    ):
        self._fnc = fnc
        self._vectorized = vectorized
        self._bulk = bulk

    @property
    def is_vectorized(self) -> bool:
        """
        Returns `True` if the function is Numba-vectorized.
        """
        return self._vectorized

    @property
    def is_bulk(self) -> bool:
        """
        Returns `True` if the function accepts inputs in 'bulk' mode.
        """
        return self._bulk

    def __call__(self, *args, **kwargs):
        return self._fnc(*args, **kwargs)
