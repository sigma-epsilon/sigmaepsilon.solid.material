from typing import Optional
import math

import numpy as np
from numpy import ndarray
from numba import guvectorize, float64

from ..config import __has_numba_cuda__

__cache = True

__all__ = [
    "principal_stress_angle_2d",
    "max_principal_stress_2d",
    "min_principal_stress_2d",
]


@guvectorize(
    [(float64, float64, float64, float64[:])],
    "(),(),()->()",
    nopython=True,
    target="cpu",
    cache=__cache,
)
def _principal_stress_angle_2d_cpu(s11, s22, s12, res):
    """
    Returns principal angles for a planar state of stress in the 1-2 plane.
    """
    res[0] = 0.5 * np.arctan(2 * s12 / (s11 - s22))


@guvectorize(
    [(float64, float64, float64, float64[:])],
    "(),(),()->()",
    nopython=True,
    target="cpu",
    cache=__cache,
)
def _max_principal_stress_2d_cpu(s11, s22, s12, res):
    """
    Returns principal angles for a planar state of stress in the 1-2 plane.
    """
    res[0] = 0.5 * (s11 + s22) + np.sqrt((0.5 * (s11 - s22)) ** 2 + s12**2)


@guvectorize(
    [(float64, float64, float64, float64[:])],
    "(),(),()->()",
    nopython=True,
    target="cpu",
    cache=__cache,
)
def _min_principal_stress_2d_cpu(s11, s22, s12, res):
    """
    Returns principal angles for a planar state of stress in the 1-2 plane.
    """
    res[0] = 0.5 * (s11 + s22) - np.sqrt((0.5 * (s11 - s22)) ** 2 + s12**2)


if __has_numba_cuda__:

    @guvectorize(
        [(float64, float64, float64, float64[:])],
        "(),(),()->()",
        nopython=True,
        target="cuda",
    )
    def _principal_stress_angle_2d_cuda(s11, s22, s12, res):
        """
        Returns principal angles for a planar state of stress in the 1-2 plane.
        """
        res[0] = 0.5 * math.atan(2 * s12 / (s11 - s22))

    @guvectorize(
        [(float64, float64, float64, float64[:])],
        "(),(),()->()",
        nopython=True,
        target="cuda",
    )
    def _max_principal_stress_2d_cuda(s11, s22, s12, res):
        """
        Returns principal angles for a planar state of stress in the 1-2 plane.
        """
        res[0] = 0.5 * (s11 + s22) + math.sqrt((0.5 * (s11 - s22)) ** 2 + s12**2)

    @guvectorize(
        [(float64, float64, float64, float64[:])],
        "(),(),()->()",
        nopython=True,
        target="cuda",
    )
    def _min_principal_stress_2d_cuda(s11, s22, s12, res):
        """
        Returns principal angles for a planar state of stress in the 1-2 plane.
        """
        res[0] = 0.5 * (s11 + s22) - math.sqrt((0.5 * (s11 - s22)) ** 2 + s12**2)

else:  # pragma: no cover

    def _principal_stress_angle_2d_cuda(*args, **kwargs):
        raise ImportError("This requires the cuda toolkit to be installed.")

    def _max_principal_stress_2d_cuda(*args, **kwargs):
        raise ImportError("This requires the cuda toolkit to be installed.")

    def _min_principal_stress_2d_cuda(*args, **kwargs):
        raise ImportError("This requires the cuda toolkit to be installed.")


def principal_stress_angle_2d(
    s11, s22=None, s12=None, device: Optional[str] = "cpu"
) -> ndarray:
    if device == "cpu":
        return _principal_stress_angle_2d_cpu(s11, s22, s12)
    elif device == "cuda":
        return _principal_stress_angle_2d_cuda(s11, s22, s12)
    else:  # pragma: no cover
        raise ValueError("The argument 'device' must be 'cpu' or 'cuda'.")


def max_principal_stress_2d(
    s11, s22=None, s12=None, device: Optional[str] = "cpu"
) -> ndarray:
    if device == "cpu":
        return _max_principal_stress_2d_cpu(s11, s22, s12)
    elif device == "cuda":
        return _max_principal_stress_2d_cuda(s11, s22, s12)
    else:  # pragma: no cover
        raise ValueError("The argument 'device' must be 'cpu' or 'cuda'.")


def min_principal_stress_2d(
    s11, s22=None, s12=None, device: Optional[str] = "cpu"
) -> ndarray:
    if device == "cpu":
        return _min_principal_stress_2d_cpu(s11, s22, s12)
    elif device == "cuda":
        return _min_principal_stress_2d_cuda(s11, s22, s12)
    else:  # pragma: no cover
        raise ValueError("The argument 'device' must be 'cpu' or 'cuda'.")
