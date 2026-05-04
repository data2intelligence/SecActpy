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
        # ridge_cuda_sparse — present in v0.2+ of libridgecuda_native.so
        # (RidgeCuda commit 5eb3130). Older builds are dense-only; the
        # sparse routing layer below detects this and falls back to CuPy.
        if hasattr(_lib, "ridge_cuda_sparse"):
            _lib.ridge_cuda_sparse.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # X
                ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),  # Y_vals
                ctypes.POINTER(ctypes.c_int),     # Y_row_indices (CSC i)
                ctypes.POINTER(ctypes.c_int),     # Y_col_pointers (CSC p)
                ctypes.c_int, ctypes.c_int,
                ctypes.c_double, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),  # beta
                ctypes.POINTER(ctypes.c_double),  # se
                ctypes.POINTER(ctypes.c_double),  # zscore
                ctypes.POINTER(ctypes.c_double),  # pvalue
                ctypes.POINTER(ctypes.c_int),     # perm_table
            ]
            _lib.ridge_cuda_sparse.restype = ctypes.c_int
        # Optional fast srand-based inverse perm-table builder (not in
        # older builds of libridgecuda_native.so).
        if hasattr(_lib, "build_inv_perm_table_srand"):
            _lib.build_inv_perm_table_srand.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int, ctypes.c_int, ctypes.c_uint,
            ]
            _lib.build_inv_perm_table_srand.restype = ctypes.c_int
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


def build_inv_perm_table_srand(n: int, n_rand: int, seed: int = 0) -> np.ndarray:
    """Fast C-side Fisher-Yates inverse permutation table generator.

    Returns (n_rand, n) int32 ndarray. Bit-equivalent to
    SecActpy's CStdlibRNG.inverse_permutation_table at the same seed —
    same algorithm (C stdlib srand+rand) — but ~200× faster (~50 ms vs
    ~11 s at n=8141, n_rand=1000) since it skips the Python interpreter
    loop. Falls back to a RuntimeError if the bundled .so doesn't export
    the symbol (older builds).
    """
    if not CUDA_NATIVE_AVAILABLE or not hasattr(_lib, "build_inv_perm_table_srand"):
        raise RuntimeError("build_inv_perm_table_srand not available in "
                           "libridgecuda_native.so. Rebuild with the "
                           "perm_helper.c addition.")
    out = np.zeros((n_rand, n), dtype=np.int32, order='C')
    rc = _lib.build_inv_perm_table_srand(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n), ctypes.c_int(n_rand), ctypes.c_uint(seed),
    )
    if rc != 0:
        raise RuntimeError(f"build_inv_perm_table_srand returned {rc}")
    return out


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


def has_sparse_kernel() -> bool:
    """True if the loaded libridgecuda_native.so exposes ridge_cuda_sparse.
    Older builds are dense-only; callers should fall back to CuPy in that
    case."""
    return CUDA_NATIVE_AVAILABLE and hasattr(_lib, "ridge_cuda_sparse")


def ridge_sparse(X, Y_csc, lambda_, n_rand, *,
                 inv_perm_table, batch_size=0, device_id=0):
    """Run RidgeCuda's compiled CUDA kernel on dense X, sparse CSC Y.

    Routes a scipy.sparse CSC matrix straight to the cusparseSpMM-based
    `ridge_cuda_sparse` symbol (RidgeCuda v0.2+). No host-side densify;
    CSR/CSC components live on the GPU as separate `data` / `indices` /
    `indptr` arrays and each per-permutation X' Y_perm is computed via
    cusparseSpMM. The same caller-supplied inv_perm_table seam used by
    the dense path applies here, so β/SE/z/p match the dense path
    bit-for-bit (within cuSPARSE-vs-cuBLAS reduction-order ε).

    Parameters
    ----------
    X : (n_genes, n_features) float64 ndarray, pre-scaled.
    Y_csc : scipy.sparse CSC matrix, shape (n_genes, n_samples).
    lambda_ : float
    n_rand : int
    inv_perm_table : (n_rand, n_genes) int32 ndarray, REQUIRED.
    """
    if not has_sparse_kernel():
        raise RuntimeError(
            "ridge_cuda_sparse not available in libridgecuda_native.so. "
            "Rebuild against RidgeCuda v0.2+ (commit 5eb3130 or later) "
            "from ridge-bench/backends/cuda_native.")
    from scipy import sparse as sps
    if not sps.issparse(Y_csc):
        raise TypeError("Y_csc must be a scipy.sparse matrix")
    Y_csc = Y_csc.tocsc().astype(np.float64)
    _ensure_init(device_id)

    X = np.asfortranarray(X, dtype=np.float64)
    n_genes, n_features = X.shape
    n_samples = Y_csc.shape[1]
    nnz = Y_csc.nnz
    if Y_csc.shape[0] != n_genes:
        raise ValueError(f"Y rows ({Y_csc.shape[0]}) != X rows ({n_genes})")

    Y_vals = np.ascontiguousarray(Y_csc.data,    dtype=np.float64)
    Y_idx  = np.ascontiguousarray(Y_csc.indices, dtype=np.int32)
    Y_ptr  = np.ascontiguousarray(Y_csc.indptr,  dtype=np.int32)

    perm = np.ascontiguousarray(inv_perm_table, dtype=np.int32)
    if perm.shape != (n_rand, n_genes):
        raise ValueError(
            f"inv_perm_table must be ({n_rand}, {n_genes}); got {perm.shape}")

    beta   = np.zeros((n_features, n_samples), dtype=np.float64, order='F')
    se     = np.zeros((n_features, n_samples), dtype=np.float64, order='F')
    zscore = np.zeros((n_features, n_samples), dtype=np.float64, order='F')
    pvalue = np.zeros((n_features, n_samples), dtype=np.float64, order='F')

    rc = _lib.ridge_cuda_sparse(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_genes), ctypes.c_int(n_features),
        Y_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        Y_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n_samples), ctypes.c_int(nnz),
        ctypes.c_double(lambda_), ctypes.c_int(n_rand), ctypes.c_int(batch_size),
        beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        se.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zscore.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pvalue.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    if rc != 0:
        raise RuntimeError(f"ridge_cuda_sparse returned status {rc}")
    return {"beta": beta, "se": se, "zscore": zscore, "pvalue": pvalue,
            "method": "cuda_native_sparse"}
