import numpy as np
from numpy import ndarray
from numba import njit, vectorize, prange

__cache = True


@njit(nogil=True, cache=__cache)
def HMH_M(strs: ndarray) -> float:
    """
    Evaluates the Huber-Mises-Hencky formula for membranes at a single point.

    Example
    -------
    >>> from sigmaepsilon.solid.material.utils import HMH_M
    >>> HMH_M((1.0, 0.0, 0.0))
    1.0
    
    Parameters
    ----------
    strs: numpy.ndarray
        The stresses s11, s22, s12.
    """
    s11, s22, s12 = strs
    return np.sqrt(s11 ** 2 - s11 * s22 + s22 ** 2 + 3 * s12 ** 2)


@njit(nogil=True, cache=__cache)
def HMH_M_multi(strs: ndarray) -> ndarray:
    """
    Evaluates the Huber-Mises-Hencky formula for membranes at several points.

    Parameters
    ----------
    strs: numpy.ndarray
        2d array of stresses for several points. The stresses 
        in the rows are expected in the order s11, s22, s12.
        
    Example
    -------
    >>> from sigmaepsilon.solid.material.utils import HMH_M_multi
    >>> import numpy as np
    >>> HMH_M_multi(np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]]))
    array([1., 1.])
    """
    nP = strs.shape[0]
    res = np.zeros(nP, dtype=strs.dtype)
    for i in prange(nP):
        res[i] = HMH_M(strs[i])
    return res


@vectorize("f8(f8, f8, f8)", target="parallel", cache=__cache)
def HMH_M_v(s11, s22, s12):
    """
    Evaluates the Huber-Mises-Hencky formula for membranes.

    Parameters
    ----------
    strs: numpy.ndarray
        The stresses s11, s22, s12.
        
    >>> from sigmaepsilon.solid.material.utils import HMH_M_v
    >>> import numpy as np
    >>> HMH_M_v(*(np.random.rand(10),)*3).shape
    (10,)
    """
    return np.sqrt(s11 ** 2 - s11 * s22 + s22 ** 2 + 3 * s12 ** 2)


@njit(nogil=True, cache=__cache)
def HMH_S(strs: ndarray) -> float:
    """
    Evaluates the Huber-Mises-Hencky formula for shells.

    Parameters
    ----------
    strs: numpy.ndarray
        The stresses s11, s22, s12, s13, s23.
        
    Example
    -------
    >>> from sigmaepsilon.solid.material.utils import HMH_S
    >>> HMH_S((1.0, 0.0, 0.0, 0.0, 0.0))
    1.0
    """
    s11, s22, s12, s13, s23 = strs
    return np.sqrt(
        s11 ** 2 - s11 * s22 + s22 ** 2 + 3 * s12 ** 2 + 3 * s13 ** 2 + 3 * s23 ** 2
    )
    
@vectorize("f8(f8, f8, f8, f8, f8)", target="parallel", cache=__cache)
def HMH_S_v(s11, s22, s12, s13, s23) -> float:
    """
    Evaluates the Huber-Mises-Hencky formula for shells.

    Parameters
    ----------
    strs: numpy.ndarray
        The stresses s11, s22, s12, s13, s23.
        
    Example
    -------
    >>> from sigmaepsilon.solid.material.utils import HMH_S_v
    >>> HMH_S_v(1.0, 0.0, 0.0, 0.0, 0.0)
    1.0
    """
    return np.sqrt(
        s11 ** 2 - s11 * s22 + s22 ** 2 + 3 * s12 ** 2 + 3 * s13 ** 2 + 3 * s23 ** 2
    )

@njit(nogil=True, cache=__cache)
def HMH_S_multi(strs: ndarray) -> ndarray:
    """
    Evaluates the Huber-Mises-Hencky formula for membranes at several points.

    Parameters
    ----------
    strs: numpy.ndarray
        2d array of stresses for several points. The stresses 
        in the rows are expected in the order s11, s22, s12, s13, s23.
        
    Example
    -------
    >>> from sigmaepsilon.solid.material.utils import HMH_S_multi
    >>> import numpy as np
    >>> HMH_S_multi(np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0]]))
    array([1., 1.])
    """
    nP = strs.shape[0]
    res = np.zeros(nP, dtype=strs.dtype)
    for i in prange(nP):
        res[i] = HMH_S(strs[i])
    return res


@njit(nogil=True, cache=__cache)
def HMH_3d(strs: ndarray) -> float:
    """
    Evaluates the Huber-Mises-Hencky formula for 3d solids.

    Parameters
    ----------
    strs: numpy.ndarray
        The stresses s11, s22, s33, s23, s13, s12.
    """
    s11, s22, s33, s23, s13, s12 = strs
    return np.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )


@njit(nogil=True, cache=__cache)
def HMH_3d_multi(strs: ndarray) -> ndarray:
    """
    Evaluates the Huber-Mises-Hencky formula for 3d solids for
    multiple points.

    Parameters
    ----------
    strs: numpy.ndarray
        2d array of stresses for several points. The stresses are
        expected in the order s11, s22, s33, s23, s13, s12.
    """
    nP = strs.shape[0]
    res = np.zeros(nP, dtype=strs.dtype)
    for i in prange(nP):
        res[i] = HMH_3d(strs[i])
    return res


@vectorize("f8(f8, f8, f8, f8, f8, f8)", target="parallel", cache=__cache)
def HMH_3d_v(s11, s22, s33, s23, s13, s12):
    """
    Vectorized evaluation of the HMH failure criterion.

    The input values s11, s22, s33, s23, s13, s12 can be
    arbitrary dimensional arrays.
    """
    return np.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )
