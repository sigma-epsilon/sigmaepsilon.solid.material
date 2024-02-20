try:  # pragma: no cover
    import cupy

    __has_cupy__ = True
except Exception:
    __has_cupy__ = False


try:  # pragma: no cover
    from numba import cuda, vectorize
    import math

    @vectorize("f8(f8, f8, f8)", target="cuda")
    def _numba_cuda_test_function(s11, s22, s12):
        return math.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)

    _numba_cuda_test_function(1.0, 1.0, 1.0)

    __has_numba_cuda__ = True
except Exception:
    __has_numba_cuda__ = False
