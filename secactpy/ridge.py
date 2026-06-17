"""Thin re-export of :func:`flashreg.ridge` for backward compatibility.

Historically, this module shipped a fork of the ridge regression
implementation. As of SecActpy v0.4, the actual kernel lives in
the :mod:`flashreg` package — same backends (numpy, omp, numba,
cupy, cuda_native), same MT19937 permutation seam, same API.

Existing code that does ``from secactpy.ridge import ridge`` continues
to work; the call now routes through the shared FlashReg library and
gets the new ``omp`` backend (C+OpenMP CPU path bit-equivalent to
NumPy) for free.

If you want to use the new backend explicitly:

    >>> from flashreg import ridge
    >>> result = ridge(X, Y, lambda_=5e5, n_rand=1000, backend='omp')
"""
from flashreg.ridge import (
    ridge,
    ridge_with_precomputed_T,
    compute_projection_matrix,
    resolve_backend,
    CUPY_AVAILABLE,
    CUDA_NATIVE_AVAILABLE,
    CUPY_INIT_ERROR,
    EPS,
    DEFAULT_LAMBDA,
    DEFAULT_NRAND,
    DEFAULT_SEED,
    _free_gpu_memory,
    _get_rng,
)

# Some legacy SecActpy code imports OMP_NATIVE_AVAILABLE / NUMBA_AVAILABLE
# directly; re-export them too if the underlying flashreg build exposes
# them (added in flashreg 0.1.0).
try:
    from flashreg.ridge import OMP_NATIVE_AVAILABLE
except ImportError:
    OMP_NATIVE_AVAILABLE = False

try:
    from flashreg.ridge import NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False

__all__ = [
    "ridge",
    "ridge_with_precomputed_T",
    "compute_projection_matrix",
    "resolve_backend",
    "CUPY_AVAILABLE",
    "CUDA_NATIVE_AVAILABLE",
    "CUPY_INIT_ERROR",
    "OMP_NATIVE_AVAILABLE",
    "NUMBA_AVAILABLE",
    "EPS",
    "DEFAULT_LAMBDA",
    "DEFAULT_NRAND",
    "DEFAULT_SEED",
    "_free_gpu_memory",
    "_get_rng",
]
