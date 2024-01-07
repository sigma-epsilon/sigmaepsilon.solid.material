try:  # pragma: no cover
    import cupy

    __has_cupy__ = True
except Exception:
    __has_cupy__ = True


try:  # pragma: no cover
    from numba import cuda

    __has_numba_cuda__ = True
except Exception:
    __has_numba_cuda__ = True
