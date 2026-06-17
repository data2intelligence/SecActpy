"""Thin re-export of :mod:`flashreg._cuda` for backward compatibility.

Historically, this module shipped a fork of the CUDA native ctypes
wrapper. As of SecActpy v0.4, that code lives in the :mod:`flashreg`
package — same ABI, same .so, same env-var fallback. The vendored
library is still located via ``SECACTPY_CUDA_NATIVE_LIB`` (legacy)
or the new ``FLASHREG_CUDA_LIB`` env var.

Drop-in equivalents:

    >>> from secactpy._cuda_native import ridge_dense
    # ... is equivalent to:
    >>> from flashregpy._cuda import ridge_dense
"""
from flashregpy._cuda import (
    ridge_dense,
    build_inv_perm_table_srand,
    CUDA_NATIVE_AVAILABLE,
)

try:
    from flashregpy._cuda import (
        ridge_sparse,
        has_sparse_kernel,
    )
except ImportError:
    ridge_sparse = None

    def has_sparse_kernel():
        return False

try:
    from flashregpy._cuda import (
        ridge_dense_yrow,
        has_yrow_kernel,
    )
except ImportError:
    ridge_dense_yrow = None

    def has_yrow_kernel():
        return False

__all__ = [
    "ridge_dense",
    "ridge_dense_yrow",
    "ridge_sparse",
    "build_inv_perm_table_srand",
    "has_sparse_kernel",
    "has_yrow_kernel",
    "CUDA_NATIVE_AVAILABLE",
]
