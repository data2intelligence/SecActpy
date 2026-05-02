"""ctypes wrapper around RidgeCuda's compiled CUDA kernel.

Provides a much faster GPU dense path than CuPy: ~14× on small-m fixtures
(GSE100093 dense: 0.84 s vs 11.7 s) by avoiding the Python+CuPy per-iter
kernel-launch dispatch and shipping the entire 1000-perm sweep to GPU
in one cudaLaunchKernel.

Bit-equivalent to the CuPy backend on β / SE / z / p when both are
given the same inverse permutation table — which they are, because both
generate the table via the same SecActpy CStdlibRNG / GSLRNG class.

The shared library `libridgecuda_native.so` is vendored at
`secactpy/_libs/libridgecuda_native.so`; build it from
`ridge-bench/backends/cuda_native` (Makefile included) and copy if
rebuilding for a new CUDA toolkit / arch.
"""
from __future__ import annotations

import ctypes
import os
import pathlib
import numpy as np

CUDA_NATIVE_AVAILABLE = False
_lib = None
_initialized = False
_init_error = None


def _find_library():
    # Priority: env var > vendored path > ridge-bench build dir.
    env = os.environ.get("SECACTPY_CUDA_NATIVE_LIB")
    if env and pathlib.Path(env).exists():
        return pathlib.Path(env)
    here = pathlib.Path(__file__).resolve().parent
    vendored = here / "_libs" / "libridgecuda_native.so"
    if vendored.exists():
        return vendored
    bench = pathlib.Path(
        "/vf/users/parks34/projects/1ridgesig/ridge-bench/backends/cuda_native"
        "/libridgecuda_native.so")
    if bench.exists():
        return bench
    return None


try:
    _path = _find_library()
    if _path is not None:
        _lib = ctypes.CDLL(str(_path))
        _lib.ridge_cuda_init.argtypes  = [ctypes.c_int]
        _lib.ridge_cuda_init.restype   = ctypes.c_int
        _lib.ridge_cuda_dense.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        _lib.ridge_cuda_dense.restype  = ctypes.c_int
        CUDA_NATIVE_AVAILABLE = True
except Exception as e:
    _init_error = e
    CUDA_NATIVE_AVAILABLE = False


def _ensure_init(device_id=0):
    global _initialized
    if _initialized:
        return
    rc = _lib.ridge_cuda_init(ctypes.c_int(device_id))
    if rc != 0:
        raise RuntimeError(f"ridge_cuda_init returned status {rc}")
    _initialized = True


def ridge_dense(X, Y, lambda_, n_rand, *,
                inv_perm_table, batch_size=0, device_id=0):
    """Run RidgeCuda's compiled CUDA kernel on dense X, dense Y.

    Parameters
    ----------
    X : (n_genes, n_features) float64 ndarray, pre-scaled.
    Y : (n_genes, n_samples) float64 ndarray, pre-scaled.
    lambda_ : float
    n_rand : int
    inv_perm_table : (n_rand, n_genes) int32 ndarray, REQUIRED — ensures
        bit-parity with SecActpy's CuPy backend (which uses the same
        inverse permutation table internally).
    """
    if not CUDA_NATIVE_AVAILABLE:
        raise RuntimeError(
            f"cuda_native unavailable: {_init_error}. Set "
            "SECACTPY_CUDA_NATIVE_LIB to point to libridgecuda_native.so.")
    _ensure_init(device_id)

    # ridge_cuda_dense expects column-major (Fortran) layout per the
    # comment at RidgeCuda/src/ridge_r_interface.cpp:301 — the .h doc
    # incorrectly says row-major.
    X = np.asfortranarray(X, dtype=np.float64)
    Y = np.asfortranarray(Y, dtype=np.float64)
    n_genes, n_features = X.shape
    n_samples = Y.shape[1]
    if Y.shape[0] != n_genes:
        raise ValueError(f"Y rows ({Y.shape[0]}) != X rows ({n_genes})")

    perm = np.ascontiguousarray(inv_perm_table, dtype=np.int32)
    if perm.shape != (n_rand, n_genes):
        raise ValueError(
            f"inv_perm_table must be ({n_rand}, {n_genes}); got {perm.shape}")

    beta   = np.zeros((n_features, n_samples), dtype=np.float64, order='F')
    se     = np.zeros((n_features, n_samples), dtype=np.float64, order='F')
    zscore = np.zeros((n_features, n_samples), dtype=np.float64, order='F')
    pvalue = np.zeros((n_features, n_samples), dtype=np.float64, order='F')

    rc = _lib.ridge_cuda_dense(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_genes), ctypes.c_int(n_features), ctypes.c_int(n_samples),
        ctypes.c_double(lambda_), ctypes.c_int(n_rand), ctypes.c_int(batch_size),
        beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        se.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zscore.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pvalue.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    if rc != 0:
        raise RuntimeError(f"ridge_cuda_dense returned status {rc}")
    return {"beta": beta, "se": se, "zscore": zscore, "pvalue": pvalue,
            "method": "cuda_native"}
