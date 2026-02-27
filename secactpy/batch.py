"""
Batch processing for large-scale ridge regression.

This module enables processing of million-sample datasets by:
1. Precomputing the projection matrix T once
2. Processing Y in memory-efficient batches
3. Optionally streaming results directly to disk (h5ad format)

Memory Management:
------------------
For a dataset with n_genes, n_features, and n_samples:
- T matrix: n_features × n_genes × 8 bytes
- Per batch: ~4 × n_features × batch_size × 8 bytes (results)
- Working memory: accumulation arrays during permutation

The `estimate_memory()` function helps determine optimal batch size.

Usage:
------
    >>> from secactpy.batch import ridge_batch, estimate_batch_size
    >>>
    >>> # Estimate optimal batch size for available memory
    >>> batch_size = estimate_batch_size(n_genes=20000, n_features=50,
    ...                                   available_gb=8.0)
    >>>
    >>> # Run batch processing (works with dense or sparse Y)
    >>> result = ridge_batch(X, Y, batch_size=batch_size)
    >>>
    >>> # Or stream directly to disk
    >>> ridge_batch(X, Y, batch_size=5000, output_path="results.h5ad")
"""

import numpy as np
from scipy import linalg
from scipy import sparse as sps
from typing import Optional, Literal, Any, Callable, Union
from dataclasses import dataclass
import time
import warnings
import gc
import math

from .rng import (
    generate_inverse_permutation_table_fast,
    get_cached_inverse_perm_table,
)
from .ridge import CUPY_AVAILABLE, EPS, DEFAULT_LAMBDA, DEFAULT_NRAND, DEFAULT_SEED, _get_rng

# Try to import h5py for streaming output
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# CuPy setup
cp = None
if CUPY_AVAILABLE:
    try:
        import cupy as cp
    except ImportError:
        pass

__all__ = [
    'ridge_batch',
    'estimate_batch_size',
    'estimate_memory',
    'StreamingResultWriter',
]


# =============================================================================
# Memory Estimation
# =============================================================================

def estimate_memory(
    n_genes: int,
    n_features: int,
    n_samples: int,
    n_rand: int = 1000,
    batch_size: Optional[int] = None,
    include_gpu: bool = False
) -> dict[str, float]:
    """
    Estimate memory requirements for ridge regression.

    Parameters
    ----------
    n_genes : int
        Number of genes/observations.
    n_features : int
        Number of features/proteins.
    n_samples : int
        Number of samples.
    n_rand : int
        Number of permutations.
    batch_size : int, optional
        Batch size. If None, assumes full dataset.
    include_gpu : bool
        Include GPU memory estimates.

    Returns
    -------
    dict
        Memory estimates in GB.
    """
    bytes_per_float = 8  # float64

    if batch_size is None:
        batch_size = n_samples

    T_bytes = n_features * n_genes * bytes_per_float
    Y_bytes = n_genes * n_samples * bytes_per_float
    results_bytes = 4 * n_features * n_samples * bytes_per_float
    working_bytes = 3 * n_features * batch_size * bytes_per_float
    perm_bytes = n_rand * n_genes * 4
    Y_batch_bytes = n_genes * batch_size * bytes_per_float
    beta_batch_bytes = n_features * batch_size * bytes_per_float

    def to_gb(x):
        return x / (1024 ** 3)

    estimates = {
        'T_matrix': to_gb(T_bytes),
        'Y_data': to_gb(Y_bytes),
        'results': to_gb(results_bytes),
        'working': to_gb(working_bytes + perm_bytes),
        'per_batch': to_gb(Y_batch_bytes + beta_batch_bytes + working_bytes),
        'total': to_gb(T_bytes + Y_bytes + results_bytes + working_bytes + perm_bytes)
    }

    if include_gpu:
        gpu_bytes = T_bytes + Y_batch_bytes + working_bytes + beta_batch_bytes
        estimates['gpu_per_batch'] = to_gb(gpu_bytes)

    return estimates


def estimate_batch_size(
    n_genes: int,
    n_features: int,
    available_gb: float = 4.0,
    n_rand: int = 1000,
    safety_factor: float = 0.7,
    min_batch: int = 100,
    max_batch: int = 50000
) -> int:
    """
    Estimate optimal batch size given available memory.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    n_features : int
        Number of features.
    available_gb : float
        Available memory in GB.
    n_rand : int
        Number of permutations.
    safety_factor : float
        Fraction of available memory to use (0-1).
    min_batch : int
        Minimum batch size.
    max_batch : int
        Maximum batch size.

    Returns
    -------
    int
        Recommended batch size.
    """
    bytes_per_float = 8
    available_bytes = available_gb * (1024 ** 3) * safety_factor

    T_bytes = n_features * n_genes * bytes_per_float
    perm_bytes = n_rand * n_genes * 4
    fixed_bytes = T_bytes + perm_bytes

    batch_bytes = available_bytes - fixed_bytes

    if batch_bytes <= 0:
        warnings.warn(
            f"Available memory ({available_gb}GB) may be insufficient. "
            f"T matrix alone requires {T_bytes / 1e9:.2f}GB."
        )
        return min_batch

    per_sample_bytes = (
        n_genes * bytes_per_float +
        4 * n_features * bytes_per_float
    )

    batch_size = int(batch_bytes / per_sample_bytes)
    batch_size = max(min_batch, min(max_batch, batch_size))

    return batch_size


# =============================================================================
# Streaming Result Writer
# =============================================================================

class StreamingResultWriter:
    """
    Stream results directly to HDF5/h5ad file.

    Writes results incrementally to avoid keeping full arrays in memory.
    """

    def __init__(
        self,
        path: str,
        n_features: int,
        n_samples: int,
        feature_names: Optional[list] = None,
        sample_names: Optional[list] = None,
        compression: Optional[str] = "gzip"
    ):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for streaming output")

        self.path = path
        self.n_features = n_features
        self.n_samples = n_samples
        self.compression = compression

        # Create file and datasets
        self.file = h5py.File(path, 'w')

        # Create resizable datasets
        for name in ['beta', 'se', 'zscore', 'pvalue']:
            self.file.create_dataset(
                name,
                shape=(n_features, n_samples),
                dtype='float64',
                compression=compression,
                chunks=(n_features, min(1000, n_samples))
            )

        # Store names if provided
        if feature_names is not None:
            self.file.create_dataset(
                'feature_names',
                data=np.array(feature_names, dtype='S')
            )
        if sample_names is not None:
            self.file.create_dataset(
                'sample_names',
                data=np.array(sample_names, dtype='S')
            )

    def write_batch(self, result: dict, start_col: int):
        """Write a batch of results to file."""
        batch_size = result['beta'].shape[1]
        end_col = start_col + batch_size

        for name in ['beta', 'se', 'zscore', 'pvalue']:
            self.file[name][:, start_col:end_col] = result[name]

        self.file.flush()

    def close(self):
        """Close the file."""
        self.file.close()


# =============================================================================
# Internal: Projection Matrix Computation
# =============================================================================

def _compute_T_numpy(X: np.ndarray, lambda_: float) -> np.ndarray:
    """Compute projection matrix T = (X'X + λI)^{-1} X' using NumPy."""
    n_genes, n_features = X.shape
    XtX = X.T @ X
    XtX_reg = XtX + lambda_ * np.eye(n_features, dtype=np.float64)

    try:
        L = linalg.cholesky(XtX_reg, lower=True)
        XtX_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
    except linalg.LinAlgError:
        warnings.warn("Cholesky decomposition failed, using pseudo-inverse")
        XtX_inv = linalg.pinv(XtX_reg)

    T = XtX_inv @ X.T
    return np.ascontiguousarray(T)


def _compute_T_cupy(X_gpu, lambda_: float):
    """Compute projection matrix T using CuPy."""
    n_genes, n_features = X_gpu.shape
    XtX = X_gpu.T @ X_gpu
    XtX_reg = XtX + lambda_ * cp.eye(n_features, dtype=cp.float64)

    try:
        XtX_inv = cp.linalg.inv(XtX_reg)
    except cp.linalg.LinAlgError:
        XtX_inv = cp.linalg.pinv(XtX_reg)

    T = XtX_inv @ X_gpu.T
    return cp.ascontiguousarray(T)


# =============================================================================
# Internal: Dense Batch Processing
# =============================================================================

def _process_batch_numpy(
    T: np.ndarray,
    Y_batch: np.ndarray,
    inv_perm_table: np.ndarray,
    n_rand: int
) -> dict[str, np.ndarray]:
    """Process a single dense batch using NumPy with T-column permutation."""
    n_features = T.shape[0]
    batch_size = Y_batch.shape[1]

    T = np.ascontiguousarray(T)
    beta = T @ Y_batch

    aver = np.zeros((n_features, batch_size), dtype=np.float64)
    aver_sq = np.zeros((n_features, batch_size), dtype=np.float64)
    pvalue_counts = np.zeros((n_features, batch_size), dtype=np.float64)
    abs_beta = np.abs(beta)

    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]
        T_perm = T[:, inv_perm_idx]
        beta_perm = T_perm @ Y_batch

        pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se = np.sqrt(np.maximum(var, 0.0))
    zscore = np.where(se > EPS, (beta - mean) / se, 0.0)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)

    return {'beta': beta, 'se': se, 'zscore': zscore, 'pvalue': pvalue}


def _process_batch_cupy(
    T_gpu,
    Y_batch: np.ndarray,
    inv_perm_table: np.ndarray,
    n_rand: int
) -> dict[str, np.ndarray]:
    """Process a single dense batch using CuPy with T-column permutation."""
    n_features = T_gpu.shape[0]
    batch_size = Y_batch.shape[1]

    Y_gpu = cp.asarray(Y_batch, dtype=cp.float64)
    beta_gpu = T_gpu @ Y_gpu

    aver = cp.zeros((n_features, batch_size), dtype=cp.float64)
    aver_sq = cp.zeros((n_features, batch_size), dtype=cp.float64)
    pvalue_counts = cp.zeros((n_features, batch_size), dtype=cp.float64)
    abs_beta = cp.abs(beta_gpu)

    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]
        inv_perm_gpu = cp.asarray(inv_perm_idx, dtype=cp.intp)
        T_perm = T_gpu[:, inv_perm_gpu]
        beta_perm = T_perm @ Y_gpu

        pvalue_counts += (cp.abs(beta_perm) >= abs_beta).astype(cp.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

        del inv_perm_gpu, T_perm, beta_perm

    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se_gpu = cp.sqrt(cp.maximum(var, 0.0))
    zscore_gpu = cp.where(se_gpu > EPS, (beta_gpu - mean) / se_gpu, 0.0)
    pvalue_gpu = (pvalue_counts + 1.0) / (n_rand + 1.0)

    result = {
        'beta': cp.asnumpy(beta_gpu),
        'se': cp.asnumpy(se_gpu),
        'zscore': cp.asnumpy(zscore_gpu),
        'pvalue': cp.asnumpy(pvalue_gpu)
    }

    del Y_gpu, beta_gpu, aver, aver_sq, pvalue_counts, abs_beta, mean, var
    del se_gpu, zscore_gpu, pvalue_gpu
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return result


# =============================================================================
# Internal: Sparse-Preserving Statistics
# =============================================================================

@dataclass
class _PopulationStats:
    """Precomputed population statistics for sparse-preserving inference."""
    mu: np.ndarray           # (n_samples,) column means
    sigma: np.ndarray        # (n_samples,) column stds
    mu_over_sigma: np.ndarray  # (n_samples,) precomputed μ/σ
    n_genes: int
    row_means: Optional[np.ndarray] = None  # (n_genes,) row means, when row_center=True

    def slice(self, start: int, end: int) -> '_PopulationStats':
        """Extract stats for a column slice."""
        return _PopulationStats(
            mu=self.mu[start:end],
            sigma=self.sigma[start:end],
            mu_over_sigma=self.mu_over_sigma[start:end],
            n_genes=self.n_genes,
            row_means=self.row_means  # shared across all column slices
        )


@dataclass
class _ProjectionComponents:
    """Precomputed projection matrix components."""
    T: np.ndarray      # (n_features, n_genes)
    c: np.ndarray      # (n_features,) row sums of T
    lambda_: float
    n_features: int
    n_genes: int


def _compute_population_stats(
    Y: Union[np.ndarray, sps.spmatrix],
    ddof: int = 1,
    row_center: bool = False
) -> _PopulationStats:
    """Compute population statistics from Y (dense or sparse).

    Parameters
    ----------
    Y : array or sparse matrix, shape (n_genes, n_samples)
    ddof : int
        Degrees of freedom for variance (1 = sample variance).
    row_center : bool
        If True, compute column stats of row-centered data (Y - row_mean)
        without actually densifying Y. This allows the batch sparse path
        to incorporate row-mean centering into in-flight normalization.
    """
    is_sparse = sps.issparse(Y)
    n_genes, n_samples = Y.shape
    row_means = None

    if is_sparse and row_center:
        # Compute stats of Y' = Y - mu_r (row centered) without densifying.
        # row means: mu_r_i = mean of row i across columns
        row_sums = np.asarray(Y.sum(axis=1)).ravel()
        row_means = row_sums / n_samples

        # Column means of Y' = mu_c - grand_mean
        col_sums = np.asarray(Y.sum(axis=0)).ravel()
        mu_c = col_sums / n_genes
        grand_mean = Y.sum() / (n_genes * n_samples)
        mu = mu_c - grand_mean

        # Column variance of Y':
        # var(Y'_j) = (1/(n-1)) * [sum_i (Y_ij - mu_r_i)^2 - n * mu'_j^2]
        # sum_i (Y_ij - mu_r_i)^2 = sum_i Y_ij^2 - 2*sum_i Y_ij*mu_r_i + sum_i mu_r_i^2
        Y_sq_col_sums = np.asarray(Y.multiply(Y).sum(axis=0)).ravel()
        cross = np.asarray(Y.T @ row_means).ravel()  # sparse.T @ dense → dense
        sum_mu_r_sq = np.sum(row_means ** 2)

        numerator = Y_sq_col_sums - 2 * cross + sum_mu_r_sq - n_genes * mu ** 2
        variance = numerator / (n_genes - ddof)
        variance = np.maximum(variance, 0)

    elif is_sparse:
        # Efficient sparse computation (no row centering)
        col_sums = np.asarray(Y.sum(axis=0)).ravel()
        mu = col_sums / n_genes

        Y_sq = Y.multiply(Y)
        col_sum_sq = np.asarray(Y_sq.sum(axis=0)).ravel()

        variance = (col_sum_sq - n_genes * mu ** 2) / (n_genes - ddof)
        variance = np.maximum(variance, 0)
    else:
        Y = np.asarray(Y)
        if row_center:
            row_means = Y.mean(axis=1)
            Y_centered = Y - row_means[:, np.newaxis]
            mu = Y_centered.mean(axis=0)
            variance = Y_centered.var(axis=0, ddof=ddof)
        else:
            mu = Y.mean(axis=0)
            variance = Y.var(axis=0, ddof=ddof)

    sigma = np.sqrt(variance)
    sigma = np.where(sigma < EPS, 1.0, sigma)
    mu_over_sigma = mu / sigma

    return _PopulationStats(
        mu=mu, sigma=sigma, mu_over_sigma=mu_over_sigma,
        n_genes=n_genes, row_means=row_means
    )


def _compute_projection_components(X: np.ndarray, lambda_: float) -> _ProjectionComponents:
    """Compute projection matrix and correction vector."""
    X = np.asarray(X, dtype=np.float64)
    n_genes, n_features = X.shape

    T = _compute_T_numpy(X, lambda_)
    c = T.sum(axis=1)

    return _ProjectionComponents(T=T, c=c, lambda_=lambda_, n_features=n_features, n_genes=n_genes)


# =============================================================================
# Internal: Sparse Batch Processing Core
# =============================================================================

def _apply_normalization_numpy(
    beta_raw: np.ndarray,
    row_corr: Optional[np.ndarray],
    sigma: Optional[np.ndarray],
    correction: Optional[np.ndarray],
) -> np.ndarray:
    """Apply row-centering, column scaling, and column centering to raw projection."""
    result = beta_raw
    if row_corr is not None:
        result = result - row_corr
    if sigma is not None:
        result = result / sigma
    if correction is not None:
        result = result - correction
    return result


def _process_sparse_batch_numpy(
    T: np.ndarray,
    c: np.ndarray,
    Y_batch: Union[np.ndarray, sps.spmatrix],
    sigma: np.ndarray,
    mu_over_sigma: np.ndarray,
    inv_perm_table: np.ndarray,
    n_rand: int,
    sparse_mode: bool = False,
    row_means: Optional[np.ndarray] = None,
    col_center: bool = True,
    col_scale: bool = True,
    mu: Optional[np.ndarray] = None
) -> dict[str, np.ndarray]:
    """
    Process a sparse batch using NumPy.

    Applies in-flight normalization controlled by col_center and col_scale:
        col_center=True,  col_scale=True:  (T @ Y) / σ - c ⊗ (μ/σ)
        col_center=True,  col_scale=False: T @ Y - c ⊗ μ
        col_center=False, col_scale=True:  (T @ Y) / σ
        col_center=False, col_scale=False: T @ Y

    When sparse_mode=True and Y_batch is sparse, avoids densifying by
    using (Y.T @ T.T).T instead of T @ Y.toarray().

    When row_means is provided, also applies row-centering correction:
        beta = (T @ Y - T @ mu_r) / σ - c ⊗ (μ'/σ')
    where μ' and σ' are column stats of row-centered data.
    """
    n_features = T.shape[0]
    is_sparse = sps.issparse(Y_batch)

    # Precompute correction term based on col_center/col_scale flags
    if col_center and col_scale:
        correction = np.outer(c, mu_over_sigma)
    elif col_center:
        correction = np.outer(c, mu) if mu is not None else np.outer(c, mu_over_sigma)
    else:
        correction = None

    # Sigma for scaling (None if col_scale is disabled)
    sigma_norm = sigma if col_scale else None

    if is_sparse and sparse_mode:
        # Sparse-preserving path: avoid densifying Y_batch
        Y_csc = Y_batch if sps.isspmatrix_csc(Y_batch) else Y_batch.tocsc()
        batch_size = Y_csc.shape[1]

        # (Y.T @ T.T).T == T @ Y, but Y stays sparse
        beta_raw = np.ascontiguousarray((Y_csc.T @ T.T).T)

        if row_means is not None:
            row_corr_obs = (T @ row_means)[:, np.newaxis]
        else:
            row_corr_obs = None
        beta = _apply_normalization_numpy(beta_raw, row_corr_obs, sigma_norm, correction)

        # Permutation testing
        aver = np.zeros((n_features, batch_size), dtype=np.float64)
        aver_sq = np.zeros((n_features, batch_size), dtype=np.float64)
        pvalue_counts = np.zeros((n_features, batch_size), dtype=np.float64)
        abs_beta = np.abs(beta)

        for i in range(n_rand):
            inv_perm_idx = inv_perm_table[i]
            T_perm = T[:, inv_perm_idx]
            beta_raw_perm = np.ascontiguousarray((Y_csc.T @ T_perm.T).T)

            if row_means is not None:
                row_corr_perm = (T_perm @ row_means)[:, np.newaxis]
            else:
                row_corr_perm = None
            beta_perm = _apply_normalization_numpy(beta_raw_perm, row_corr_perm, sigma_norm, correction)

            pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
            aver += beta_perm
            aver_sq += beta_perm ** 2
    else:
        # Dense path (original behavior)
        if is_sparse:
            Y_dense = Y_batch.toarray()
        else:
            Y_dense = np.asarray(Y_batch)

        batch_size = Y_dense.shape[1]

        beta_raw = T @ Y_dense

        if row_means is not None:
            row_corr_obs = (T @ row_means)[:, np.newaxis]
        else:
            row_corr_obs = None
        beta = _apply_normalization_numpy(beta_raw, row_corr_obs, sigma_norm, correction)

        # Permutation testing
        aver = np.zeros((n_features, batch_size), dtype=np.float64)
        aver_sq = np.zeros((n_features, batch_size), dtype=np.float64)
        pvalue_counts = np.zeros((n_features, batch_size), dtype=np.float64)
        abs_beta = np.abs(beta)

        for i in range(n_rand):
            inv_perm_idx = inv_perm_table[i]
            T_perm = T[:, inv_perm_idx]
            beta_raw_perm = T_perm @ Y_dense

            if row_means is not None:
                row_corr_perm = (T_perm @ row_means)[:, np.newaxis]
            else:
                row_corr_perm = None
            beta_perm = _apply_normalization_numpy(beta_raw_perm, row_corr_perm, sigma_norm, correction)

            pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
            aver += beta_perm
            aver_sq += beta_perm ** 2

    # Finalize statistics
    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se = np.sqrt(np.maximum(var, 0.0))
    zscore = np.where(se > EPS, (beta - mean) / se, 0.0)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)

    return {'beta': beta, 'se': se, 'zscore': zscore, 'pvalue': pvalue}


def _process_sparse_batch_cupy(
    T: np.ndarray,
    c: np.ndarray,
    Y_batch: Union[np.ndarray, sps.spmatrix],
    sigma: np.ndarray,
    mu_over_sigma: np.ndarray,
    inv_perm_table: np.ndarray,
    n_rand: int,
    sparse_mode: bool = False,
    row_means: Optional[np.ndarray] = None,
    col_center: bool = True,
    col_scale: bool = True,
    mu: Optional[np.ndarray] = None
) -> dict[str, np.ndarray]:
    """Process a sparse batch using CuPy with in-flight normalization.

    Normalization is controlled by col_center and col_scale flags
    (see _process_sparse_batch_numpy for the full formula table).

    When sparse_mode=True and Y_batch is sparse, transfers Y as a CuPy
    sparse matrix and uses (Y.T @ T.T).T to avoid densifying.

    When row_means is provided, also applies row-centering correction.
    """
    import cupyx.scipy.sparse as cpsps

    n_features = T.shape[0]
    is_sparse = sps.issparse(Y_batch)

    # Transfer to GPU
    T_gpu = cp.asarray(T, dtype=cp.float64)
    c_gpu = cp.asarray(c, dtype=cp.float64)
    sigma_gpu = cp.asarray(sigma, dtype=cp.float64) if col_scale else None
    mu_over_sigma_gpu = cp.asarray(mu_over_sigma, dtype=cp.float64) if (col_center and col_scale) else None
    mu_gpu = cp.asarray(mu, dtype=cp.float64) if (col_center and not col_scale and mu is not None) else None
    row_means_gpu = cp.asarray(row_means, dtype=cp.float64) if row_means is not None else None

    use_sparse_gpu = is_sparse and sparse_mode

    if use_sparse_gpu:
        Y_csc = Y_batch if sps.isspmatrix_csc(Y_batch) else Y_batch.tocsc()
        Y_gpu = cpsps.csc_matrix(Y_csc, dtype=cp.float64)
        batch_size = Y_csc.shape[1]
    else:
        if is_sparse:
            Y_gpu = cp.asarray(Y_batch.toarray(), dtype=cp.float64)
        else:
            Y_gpu = cp.asarray(Y_batch, dtype=cp.float64)
        batch_size = Y_gpu.shape[1]

    # Precompute correction term based on col_center/col_scale flags
    if col_center and col_scale:
        correction_gpu = cp.outer(c_gpu, mu_over_sigma_gpu)
    elif col_center:
        correction_gpu = cp.outer(c_gpu, mu_gpu)
    else:
        correction_gpu = None

    # Compute beta
    if use_sparse_gpu:
        beta_raw_gpu = (Y_gpu.T @ T_gpu.T).T
    else:
        beta_raw_gpu = T_gpu @ Y_gpu

    # Apply normalization
    result_gpu = beta_raw_gpu
    if row_means_gpu is not None:
        row_corr_gpu = (T_gpu @ row_means_gpu)[:, cp.newaxis]
        result_gpu = result_gpu - row_corr_gpu
    if sigma_gpu is not None:
        result_gpu = result_gpu / sigma_gpu
    if correction_gpu is not None:
        result_gpu = result_gpu - correction_gpu
    beta_gpu = result_gpu

    # Permutation testing
    aver_gpu = cp.zeros((n_features, batch_size), dtype=cp.float64)
    aver_sq_gpu = cp.zeros((n_features, batch_size), dtype=cp.float64)
    pvalue_counts_gpu = cp.zeros((n_features, batch_size), dtype=cp.float64)
    abs_beta_gpu = cp.abs(beta_gpu)

    # Transfer permutation table to GPU
    inv_perm_table_gpu = cp.asarray(inv_perm_table, dtype=cp.int32)

    for i in range(n_rand):
        inv_perm_idx = inv_perm_table_gpu[i]
        T_perm_gpu = T_gpu[:, inv_perm_idx]

        if use_sparse_gpu:
            beta_raw_perm_gpu = (Y_gpu.T @ T_perm_gpu.T).T
        else:
            beta_raw_perm_gpu = T_perm_gpu @ Y_gpu

        perm_result_gpu = beta_raw_perm_gpu
        if row_means_gpu is not None:
            row_corr_perm_gpu = (T_perm_gpu @ row_means_gpu)[:, cp.newaxis]
            perm_result_gpu = perm_result_gpu - row_corr_perm_gpu
        if sigma_gpu is not None:
            perm_result_gpu = perm_result_gpu / sigma_gpu
        if correction_gpu is not None:
            perm_result_gpu = perm_result_gpu - correction_gpu
        beta_perm_gpu = perm_result_gpu

        pvalue_counts_gpu += (cp.abs(beta_perm_gpu) >= abs_beta_gpu).astype(cp.float64)
        aver_gpu += beta_perm_gpu
        aver_sq_gpu += beta_perm_gpu ** 2

    # Finalize statistics
    mean_gpu = aver_gpu / n_rand
    var_gpu = (aver_sq_gpu / n_rand) - (mean_gpu ** 2)
    se_gpu = cp.sqrt(cp.maximum(var_gpu, 0.0))
    zscore_gpu = cp.where(se_gpu > EPS, (beta_gpu - mean_gpu) / se_gpu, 0.0)
    pvalue_gpu = (pvalue_counts_gpu + 1.0) / (n_rand + 1.0)

    # Transfer back
    result = {
        'beta': cp.asnumpy(beta_gpu),
        'se': cp.asnumpy(se_gpu),
        'zscore': cp.asnumpy(zscore_gpu),
        'pvalue': cp.asnumpy(pvalue_gpu)
    }

    # Cleanup
    del T_gpu, c_gpu, Y_gpu
    del beta_raw_gpu, beta_gpu, inv_perm_table_gpu
    del aver_gpu, aver_sq_gpu, pvalue_counts_gpu, abs_beta_gpu
    del mean_gpu, var_gpu, se_gpu, zscore_gpu, pvalue_gpu
    if sigma_gpu is not None:
        del sigma_gpu
    if mu_over_sigma_gpu is not None:
        del mu_over_sigma_gpu
    if mu_gpu is not None:
        del mu_gpu
    if correction_gpu is not None:
        del correction_gpu
    if row_means_gpu is not None:
        del row_means_gpu
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return result


# =============================================================================
# Internal: Sparse Batch Path
# =============================================================================

def _ridge_batch_sparse_path(
    X: np.ndarray,
    Y: sps.spmatrix,
    lambda_: float,
    n_rand: int,
    seed: int,
    batch_size: int,
    backend: str,
    use_gsl_rng: bool,
    rng_method: str,
    use_cache: bool,
    output_path: Optional[str],
    output_compression: Optional[str],
    feature_names: Optional[list],
    sample_names: Optional[list],
    progress_callback: Optional[Callable[[int, int], None]],
    verbose: bool,
    start_time: float,
    sparse_mode: bool = False,
    row_center: bool = False,
    col_center: bool = True,
    col_scale: bool = True
) -> Optional[dict[str, Any]]:
    """
    Internal sparse path for ridge_batch.

    Key optimization: Computes population stats ONCE from full Y,
    then slices stats per batch instead of recomputing.
    """
    # Ensure CSC for efficient column slicing
    if not sps.isspmatrix_csc(Y):
        Y = Y.tocsc()
    
    X = np.asarray(X, dtype=np.float64)
    
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of rows: {X.shape[0]} vs {Y.shape[0]}")
    if n_rand <= 0:
        raise ValueError("Batch processing requires n_rand > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    n_genes, n_features = X.shape
    n_samples = Y.shape[1]
    n_batches = math.ceil(n_samples / batch_size)
    
    # Backend selection
    if backend == "auto":
        backend = "cupy" if CUPY_AVAILABLE else "numpy"
    elif backend == "cupy" and not CUPY_AVAILABLE:
        raise ImportError("CuPy backend requested but not available")
    
    use_gpu = (backend == "cupy")
    
    if verbose:
        sparse_mem_mb = (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / 1e6
        nnz_pct = 100 * Y.nnz / (Y.shape[0] * Y.shape[1])
        print("Ridge batch processing (sparse input):")
        print(f"  Data: {n_genes} genes, {n_features} features, {n_samples} samples")
        print(f"  Sparsity: {100-nnz_pct:.1f}% zeros ({Y.nnz:,} non-zeros)")
        print(f"  Sparse memory: {sparse_mem_mb:.1f} MB")
        print(f"  Batches: {n_batches} (size={batch_size})")
        print(f"  Backend: {backend}")
    
    # === PRECOMPUTE ONCE ===
    if verbose:
        print("  Precomputing projection matrix T...")
    t_start = time.time()
    proj = _compute_projection_components(X, lambda_)
    if verbose:
        print(f"  T matrix computed in {time.time() - t_start:.2f}s")
    
    if verbose:
        msg = "  Precomputing population statistics from full Y"
        if row_center:
            msg += " (with row centering)"
        print(msg + "...")
    t_start = time.time()
    full_stats = _compute_population_stats(Y, row_center=row_center)
    if verbose:
        print(f"  Stats computed in {time.time() - t_start:.2f}s")
    
    # Get inverse permutation table
    if verbose:
        print("  Loading inverse permutation table...")

    rng_obj, use_deterministic = _get_rng(rng_method, use_gsl_rng, seed)
    if use_deterministic:
        if use_cache:
            inv_perm_table = get_cached_inverse_perm_table(n_genes, n_rand, seed, verbose=verbose)
        else:
            inv_perm_table = rng_obj.inverse_permutation_table(n_genes, n_rand)
    else:
        if verbose:
            print("  Generating permutation table (fast NumPy RNG)...")
        inv_perm_table = generate_inverse_permutation_table_fast(n_genes, n_rand, seed)

    # Setup streaming output
    writer = None
    if output_path is not None:
        if verbose:
            print(f"  Output: streaming to {output_path}")
        writer = StreamingResultWriter(
            output_path,
            n_features=n_features,
            n_samples=n_samples,
            feature_names=feature_names,
            sample_names=sample_names,
            compression=output_compression
        )
    
    # === PROCESS BATCHES ===
    if verbose:
        print(f"  Processing {n_batches} batches...")
    
    results_list = [] if writer is None else None
    
    for batch_idx in range(n_batches):
        batch_start_time = time.time()
        
        # Batch column indices
        start_col = batch_idx * batch_size
        end_col = min(start_col + batch_size, n_samples)
        
        # Extract batch data (sparse slice)
        Y_batch = Y[:, start_col:end_col]
        
        # SLICE stats (not recompute!)
        batch_stats = full_stats.slice(start_col, end_col)
        
        # Process batch
        if use_gpu:
            batch_result = _process_sparse_batch_cupy(
                proj.T, proj.c, Y_batch,
                batch_stats.sigma, batch_stats.mu_over_sigma,
                inv_perm_table, n_rand,
                sparse_mode=sparse_mode,
                row_means=batch_stats.row_means,
                col_center=col_center,
                col_scale=col_scale,
                mu=batch_stats.mu
            )
        else:
            batch_result = _process_sparse_batch_numpy(
                proj.T, proj.c, Y_batch,
                batch_stats.sigma, batch_stats.mu_over_sigma,
                inv_perm_table, n_rand,
                sparse_mode=sparse_mode,
                row_means=batch_stats.row_means,
                col_center=col_center,
                col_scale=col_scale,
                mu=batch_stats.mu
            )
        
        # Store or write
        if writer is not None:
            writer.write_batch(batch_result, start_col=start_col)
        else:
            results_list.append(batch_result)
        
        # Progress callback
        if progress_callback is not None:
            progress_callback(batch_idx, n_batches)
        
        if verbose:
            batch_time = time.time() - batch_start_time
            print(f"    Batch {batch_idx + 1}/{n_batches}: {end_col - start_col} samples in {batch_time:.2f}s")
        
        # Cleanup
        del Y_batch, batch_stats, batch_result
        gc.collect()
    
    # Finalize
    total_time = time.time() - start_time
    
    del proj, full_stats, inv_perm_table
    if use_gpu and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    if writer is not None:
        writer.close()
        if verbose:
            print(f"  Results written to {output_path}")
            print(f"  Completed in {total_time:.2f}s")
        return None
    
    # Concatenate results
    if verbose:
        print("  Concatenating results...")
    
    final_result = {
        'beta': np.hstack([r['beta'] for r in results_list]),
        'se': np.hstack([r['se'] for r in results_list]),
        'zscore': np.hstack([r['zscore'] for r in results_list]),
        'pvalue': np.hstack([r['pvalue'] for r in results_list]),
        'method': f"{backend}_batch_sparse",
        'time': total_time,
        'n_batches': n_batches
    }
    
    if verbose:
        print(f"  Completed in {total_time:.2f}s")
    
    return final_result


# =============================================================================
# Main Public API
# =============================================================================

def ridge_batch(
    X: np.ndarray,
    Y: Union[np.ndarray, sps.spmatrix],
    lambda_: float = DEFAULT_LAMBDA,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    batch_size: int = 5000,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    use_gsl_rng: bool = True,
    rng_method: str = None,
    use_cache: bool = False,
    output_path: Optional[str] = None,
    output_compression: Optional[str] = "gzip",
    feature_names: Optional[list] = None,
    sample_names: Optional[list] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    sparse_mode: bool = False,
    row_center: bool = False,
    col_center: bool = True,
    col_scale: bool = True,
    verbose: bool = False
) -> Optional[dict[str, Any]]:
    """
    Ridge regression with batch processing for large datasets.

    Computes T = (X'X + λI)^{-1} X' once, then processes Y in batches.
    Handles both dense and sparse Y matrices efficiently.

    Parameters
    ----------
    X : ndarray, shape (n_genes, n_features)
        Design matrix (signature). Must be dense.
    Y : ndarray or sparse matrix, shape (n_genes, n_samples)
        Response matrix (expression). Can be dense or sparse.
        - Dense: Should be pre-scaled (mean=0, std=1 per column)
        - Sparse: Raw counts, scaling applied in-flight
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations for significance testing.
    seed : int, default=0
        Random seed for permutations.
    batch_size : int, default=5000
        Number of samples per batch.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Computation backend.
    use_gsl_rng : bool, default=True
        Use GSL-compatible RNG for exact R/RidgeR reproducibility.
        Set to False for faster inference (~70x faster permutation
        generation) when R matching is not needed.
        Ignored when ``rng_method`` is set.
    rng_method : {"srand", "gsl", "numpy", None}, default=None
        Explicit RNG backend selection. Overrides ``use_gsl_rng`` when set.

        - ``"srand"``: C stdlib srand/rand (matches R's RidgeR behavior)
        - ``"gsl"``: GSL random number generator
        - ``"numpy"``: Fast NumPy RNG (~70x faster permutations)
        - ``None``: Falls back to ``use_gsl_rng`` for backward compatibility
    use_cache : bool, default=False
        Cache permutation tables to disk for reuse.
    output_path : str, optional
        Stream results to HDF5 file instead of returning in memory.
    output_compression : str, optional
        Compression for streaming ("gzip", "lzf", or None).
    feature_names : list, optional
        Feature names for output file.
    sample_names : list, optional
        Sample names for output file.
    progress_callback : callable, optional
        Function(batch_idx, n_batches) for progress tracking.
    sparse_mode : bool, default=False
        When True and Y is a sparse matrix, avoid densifying Y batches.
        Uses (Y.T @ T.T).T instead of T @ Y.toarray(), which is
        30-40x more memory-efficient for very sparse data (<5% density).
        Trade-off: ~25% slower at 5-10% density. Default (False) keeps
        the original dense behavior.
    row_center : bool, default=False
        When True, apply row-mean centering as part of in-flight
        normalization. This computes column statistics of the
        row-centered data (Y - row_mean) without actually densifying Y.
        Used by high-level scRNAseq/ST functions to keep the full
        pipeline sparse.
    col_center : bool, default=True
        When True (default), subtract column means during in-flight
        normalization of sparse Y. Only used for sparse path; ignored
        for dense Y (which should be pre-centered).
    col_scale : bool, default=True
        When True (default), divide by column standard deviations during
        in-flight normalization of sparse Y. Only used for sparse path;
        ignored for dense Y (which should be pre-scaled).
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    dict or None
        If output_path is None, returns:
        - beta, se, zscore, pvalue: ndarrays (n_features, n_samples)
        - method, time, n_batches: metadata
        
        If output_path provided, returns None (results in file).

    Examples
    --------
    >>> # Dense input (pre-scaled)
    >>> Y_scaled = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    >>> result = ridge_batch(X, Y_scaled, batch_size=5000)
    
    >>> # Sparse input (raw counts, auto-scaled)
    >>> Y_sparse = scipy.sparse.csr_matrix(counts)
    >>> result = ridge_batch(X, Y_sparse, batch_size=10000)
    
    >>> # Stream to disk for very large datasets
    >>> ridge_batch(X, Y, batch_size=10000, output_path="results.h5ad")
    """
    start_time = time.time()
    
    # === SPARSE PATH ===
    if sps.issparse(Y):
        return _ridge_batch_sparse_path(
            X=X, Y=Y, lambda_=lambda_, n_rand=n_rand, seed=seed,
            batch_size=batch_size, backend=backend, use_gsl_rng=use_gsl_rng, rng_method=rng_method,
            use_cache=use_cache,
            output_path=output_path, output_compression=output_compression,
            feature_names=feature_names, sample_names=sample_names,
            progress_callback=progress_callback, verbose=verbose,
            start_time=start_time, sparse_mode=sparse_mode,
            row_center=row_center,
            col_center=col_center,
            col_scale=col_scale
        )
    
    # === DENSE PATH ===
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of rows: {X.shape[0]} vs {Y.shape[0]}")
    if n_rand <= 0:
        raise ValueError("Batch processing requires n_rand > 0. Use ridge() for t-test.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    n_genes, n_features = X.shape
    n_samples = Y.shape[1]
    n_batches = math.ceil(n_samples / batch_size)
    
    if verbose:
        print("Ridge batch processing:")
        print(f"  Data: {n_genes} genes, {n_features} features, {n_samples} samples")
        print(f"  Batches: {n_batches} (size={batch_size})")
        mem = estimate_memory(n_genes, n_features, n_samples, n_rand, batch_size)
        print(f"  Estimated memory: {mem['total']:.2f} GB total, {mem['per_batch']:.3f} GB per batch")
    
    # Backend selection
    if backend == "auto":
        backend = "cupy" if CUPY_AVAILABLE else "numpy"
    elif backend == "cupy" and not CUPY_AVAILABLE:
        raise ImportError("CuPy backend requested but not available")
    
    use_gpu = (backend == "cupy")
    
    if verbose:
        print(f"  Backend: {backend}")
    
    # Setup streaming output
    writer = None
    if output_path is not None:
        if verbose:
            print(f"  Output: streaming to {output_path}")
            if output_compression:
                print(f"  Compression: {output_compression}")
        writer = StreamingResultWriter(
            output_path,
            n_features=n_features,
            n_samples=n_samples,
            feature_names=feature_names,
            sample_names=sample_names,
            compression=output_compression
        )
    
    # Compute T matrix once
    if verbose:
        print("  Computing projection matrix T...")
    
    t_start = time.time()
    if use_gpu:
        X_gpu = cp.asarray(X, dtype=cp.float64)
        T = _compute_T_cupy(X_gpu, lambda_)
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    else:
        T = _compute_T_numpy(X, lambda_)
    
    if verbose:
        print(f"  T matrix computed in {time.time() - t_start:.2f}s")
    
    # Get inverse permutation table
    if verbose:
        print("  Loading inverse permutation table...")

    rng_obj, use_deterministic = _get_rng(rng_method, use_gsl_rng, seed)
    if use_deterministic:
        if use_cache:
            inv_perm_table = get_cached_inverse_perm_table(n_genes, n_rand, seed, verbose=verbose)
        else:
            inv_perm_table = rng_obj.inverse_permutation_table(n_genes, n_rand)
    else:
        if verbose:
            print("  Generating permutation table (fast NumPy RNG)...")
        inv_perm_table = generate_inverse_permutation_table_fast(n_genes, n_rand, seed)

    # Process batches
    if verbose:
        print(f"  Processing {n_batches} batches...")
    
    results_list = [] if writer is None else None
    
    for batch_idx in range(n_batches):
        batch_start = time.time()
        
        start_col = batch_idx * batch_size
        end_col = min(start_col + batch_size, n_samples)
        Y_batch = Y[:, start_col:end_col]
        
        if use_gpu:
            batch_result = _process_batch_cupy(T, Y_batch, inv_perm_table, n_rand)
        else:
            batch_result = _process_batch_numpy(T, Y_batch, inv_perm_table, n_rand)
        
        if writer is not None:
            writer.write_batch(batch_result, start_col=start_col)
        else:
            results_list.append(batch_result)
        
        if progress_callback is not None:
            progress_callback(batch_idx, n_batches)
        
        if verbose:
            batch_time = time.time() - batch_start
            print(f"    Batch {batch_idx + 1}/{n_batches}: {end_col - start_col} samples in {batch_time:.2f}s")
        
        del Y_batch, batch_result
        gc.collect()
    
    # Finalize
    total_time = time.time() - start_time
    
    del T, inv_perm_table
    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    
    if writer is not None:
        writer.close()
        if verbose:
            print(f"  Results written to {output_path}")
            print(f"  Completed in {total_time:.2f}s")
        return None
    
    if verbose:
        print("  Concatenating results...")
    
    final_result = {
        'beta': np.hstack([r['beta'] for r in results_list]),
        'se': np.hstack([r['se'] for r in results_list]),
        'zscore': np.hstack([r['zscore'] for r in results_list]),
        'pvalue': np.hstack([r['pvalue'] for r in results_list]),
        'method': f"{backend}_batch",
        'time': total_time,
        'n_batches': n_batches
    }
    
    if verbose:
        print(f"  Completed in {total_time:.2f}s")
    
    return final_result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SecActPy Batch Module - Testing")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test parameters
    n_genes = 500
    n_features = 20
    n_samples = 1000
    batch_size = 200
    n_rand = 50
    
    X = np.random.randn(n_genes, n_features)
    Y_dense = np.random.randn(n_genes, n_samples)
    
    # Scale for dense path
    Y_scaled = (Y_dense - Y_dense.mean(0)) / Y_dense.std(0, ddof=1)
    
    # Create sparse version (raw)
    Y_sparse = sps.csr_matrix(Y_dense)
    
    print(f"\nTest data: X({n_genes}, {n_features}), Y({n_genes}, {n_samples})")
    print(f"batch_size={batch_size}, n_rand={n_rand}")
    
    # Test 1: Dense batch
    print("\n1. Testing dense batch processing...")
    result_dense = ridge_batch(
        X, Y_scaled,
        lambda_=5e5, n_rand=n_rand, seed=0,
        batch_size=batch_size, backend='numpy', verbose=True
    )
    print(f"   Result shape: {result_dense['beta'].shape}")
    
    # Test 2: Sparse batch
    print("\n2. Testing sparse batch processing...")
    result_sparse = ridge_batch(
        X, Y_sparse,
        lambda_=5e5, n_rand=n_rand, seed=0,
        batch_size=batch_size, backend='numpy', verbose=True
    )
    print(f"   Result shape: {result_sparse['beta'].shape}")
    
    # Test 3: Compare results
    print("\n3. Comparing dense vs sparse results...")
    for key in ['beta', 'se', 'zscore', 'pvalue']:
        diff = np.abs(result_dense[key] - result_sparse[key]).max()
        status = "✓" if diff < 1e-10 else "✗"
        print(f"   {status} {key}: max diff = {diff:.2e}")
    
    # Test 4: Compare with standard ridge
    print("\n4. Comparing with standard ridge...")
    from .ridge import ridge
    result_std = ridge(X, Y_scaled, lambda_=5e5, n_rand=n_rand, seed=0, backend='numpy')
    
    for key in ['beta', 'se', 'zscore', 'pvalue']:
        diff = np.abs(result_dense[key] - result_std[key]).max()
        status = "✓" if diff < 1e-10 else "✗"
        print(f"   {status} {key}: max diff = {diff:.2e}")
    
    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
