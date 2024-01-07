from typing import Union, Iterable
import math

import numpy as np
from numpy import ndarray
from numba import njit, vectorize, guvectorize, prange, float64

from ..config import __has_cupy__, __has_numba_cuda__

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
    return np.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)


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
    Evaluates the Huber-Mises-Hencky formula for membranes on the cpu
    using high-level parallelization.

    Parameters
    ----------
    strs: numpy.ndarray
        The stresses s11, s22, s12.

    >>> from sigmaepsilon.solid.material.utils import HMH_M_v
    >>> import numpy as np
    >>> HMH_M_v(*(np.random.rand(10),)*3).shape
    (10,)
    """
    return np.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)


@vectorize("f8(f8, f8, f8)", target="cuda")
def HMH_M_v_cuda(s11, s22, s12):
    """
    Evaluates the Huber-Mises-Hencky formula for membranes on the gpu.

    Parameters
    ----------
    strs: numpy.ndarray
        The stresses s11, s22, s12.

    >>> from sigmaepsilon.solid.material.utils import HMH_M_v
    >>> import numpy as np
    >>> HMH_M_v(*(np.random.rand(10),)*3).shape
    (10,)
    """
    return math.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)


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
        s11**2 - s11 * s22 + s22**2 + 3 * s12**2 + 3 * s13**2 + 3 * s23**2
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
        s11**2 - s11 * s22 + s22**2 + 3 * s12**2 + 3 * s13**2 + 3 * s23**2
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
        + 3 * (s12**2 + s13**2 + s23**2)
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
        + 3 * (s12**2 + s13**2 + s23**2)
    )


@vectorize("f8(f8, f8, f8, f8, f8, f8)", target="cuda")
def HMH_3d_v_cuda(s11, s22, s33, s23, s13, s12):
    """
    Vectorized evaluation of the HMH failure criterion.

    The input values s11, s22, s33, s23, s13, s12 can be
    arbitrary dimensional arrays.
    """
    return math.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3 * (s12**2 + s13**2 + s23**2)
    )


@guvectorize(
    [(float64, float64, float64, float64, float64, float64, float64[:])],
    "(),(),(),(),(),()->()",
    target="cuda",
)
def HMH_3d_guv_cuda(s11, s22, s33, s23, s13, s12, output):
    """
    Vectorized evaluation of the HMH failure criterion.

    The input values s11, s22, s33, s23, s13, s12 can be
    arbitrary dimensional arrays.
    """
    output[0] = math.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3 * (s12**2 + s13**2 + s23**2)
    )


def HMH_3d_v_cuda_cp(s11, s22, s33, s23, s13, s12):
    raise NotImplementedError(
        (
            "You need to install CuPy and the proper version of the cuda toolkit"
            " or choose another device."
        )
    )


if __has_numba_cuda__:
    from numba import cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    @cuda.jit
    def HMH_3d_v_cuda_kernel(s11, s22, s33, s23, s13, s12, output):
        idx = cuda.grid(1)
        if idx < s11.size:
            output[idx] = math.sqrt(
                0.5
                * (
                    (s11[idx] - s22[idx]) ** 2
                    + (s22[idx] - s33[idx]) ** 2
                    + (s33[idx] - s11[idx]) ** 2
                )
                + 3 * (s12[idx] ** 2 + s13[idx] ** 2 + s23[idx] ** 2)
            )
            
    @cuda.jit
    def _divide_array(arr, divisor):
        idx = cuda.grid(1)
        if idx < arr.size:
            arr[idx] = arr[idx] / divisor

    def _to_device_array(arr: Union[ndarray, DeviceNDArray]):
        if isinstance(arr, DeviceNDArray):
            return arr
        elif isinstance(arr, ndarray):
            device_array = cuda.to_device(arr)
            return device_array
        else:
            raise TypeError

    def _are_all_numba_device_arrays(arrays: Iterable[ndarray]):
        return all(map(lambda arr: isinstance(arr, DeviceNDArray), arrays))

    def divide_array(arr, divisor, threads_per_block=512, blocks_per_grid=None):        
        if blocks_per_grid is None:
            blocks_per_grid = int(np.ceil(len(arr) / threads_per_block))

        kernel = _divide_array[blocks_per_grid, threads_per_block]
        kernel(arr, divisor)
    
    def HMH_3d_v_numba_cuda_kernel(
        *stresses, threads_per_block=512, blocks_per_grid=None
    ) -> ndarray:
        all_device = _are_all_numba_device_arrays(stresses)
        
        if not all_device:
            device_stresses = tuple(map(lambda s: _to_device_array(s), stresses))
        else:
            device_stresses = stresses
            
        output_device = cuda.device_array(stresses[0].shape[0], dtype=np.float64)

        if blocks_per_grid is None:
            blocks_per_grid = int(np.ceil(stresses[0].shape[0] / threads_per_block))

        kernel = HMH_3d_v_cuda_kernel[blocks_per_grid, threads_per_block]
        kernel(*device_stresses, output_device)

        if all_device:
            output = output_device
        else:
            output = output_device.copy_to_host()

        return output
else:
    def divide_array(arr, divisor) -> ndarray:
        arr /= divisor
        return arr


if __has_cupy__:
    import cupy as cp
    
    def _are_all_cupy_device_arrays(arrays: Iterable[ndarray]):
        return all(map(lambda arr: isinstance(arr, cp.ndarray), arrays))

    def HMH_3d_v_cuda_cp(s11, s22, s33, s23, s13, s12):
        """
        Vectorized evaluation of the HMH failure criterion on the gpu using CuPy.

        The input values s11, s22, s33, s23, s13, s12 can be
        arbitrary dimensional arrays.
        """
        stresses = s11, s22, s33, s23, s13, s12
        all_device = _are_all_cupy_device_arrays(stresses)
        
        if not all_device:
            s11, s22, s33, s23, s13, s12 = tuple(
                map(lambda s: cp.asarray(s), (s11, s22, s33, s23, s13, s12))
            )
        
        cupy_array = cp.sqrt(
            0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
            + 3 * (s12**2 + s13**2 + s23**2)
        )
        
        if not all_device:
            return cp.asnumpy(cupy_array)
        else:
            return cupy_array
