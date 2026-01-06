"""
Ridge regression with permutation-based significance testing.

This module provides the core ridge regression algorithm with permutation
testing for significance, matching RidgeR's output exactly.

Algorithm:
----------
1. Compute projection matrix: T = (X'X + λI)^{-1} X'
2. Compute coefficients: β = T @ Y
3. For each permutation:
   - Permute rows of Y (or equivalently, columns of T)
   - Compute β_rand = T @ Y_perm
   - Accumulate statistics
4. Compute SE, z-score, p-value from permutation distribution

The permutation p-value is computed as:
    p = (count(|β_rand| >= |β_obs|) + 1) / (n_rand + 1)

This matches RidgeR's implementation exactly when using the same seed.

Usage:
------
    >>> from secactpy.ridge import ridge
    >>>
    >>> # Basic usage
    >>> result = ridge(X, Y, lambda_=5e5, n_rand=1000)
    >>>
    >>> # Access results
    >>> beta = result['beta']       # Coefficients
    >>> pvalue = result['pvalue']   # P-values
    >>> zscore = result['zscore']   # Z-scores
    >>>
    >>> # GPU acceleration
    >>> result = ridge(X, Y, lambda_=5e5, n_rand=1000, backend='cupy')
"""

import numpy as np
from scipy import linalg
from scipy import stats
from typing import Literal, Any
import time
import warnings
import gc

from .rng import (
    GSLRNG,
    generate_inverse_permutation_table_fast,
    get_cached_inverse_perm_table,
)

__all__ = ['ridge', 'CUPY_AVAILABLE']


# =============================================================================
# CuPy Setup
# =============================================================================

CUPY_AVAILABLE = False
CUPY_INIT_ERROR = None
cp = None

try:
    import cupy as cp
    # Test GPU availability
    _ = cp.array([1.0])
    cp.cuda.Device().synchronize()
    CUPY_AVAILABLE = True
except ImportError:
    pass
except Exception as e:
    # Store the error but don't warn yet - only warn when GPU is actually requested
    CUPY_INIT_ERROR = str(e)


# =============================================================================
# Constants
# =============================================================================

# Tolerance for near-zero standard deviation
EPS = 1e-12

# Default parameters (matching RidgeR)
DEFAULT_LAMBDA = 5e5
DEFAULT_NRAND = 1000
DEFAULT_SEED = 0


# =============================================================================
# Main Ridge Function
# =============================================================================

def ridge(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_: float = DEFAULT_LAMBDA,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    use_gsl_rng: bool = True,
    use_cache: bool = False,
    verbose: bool = False
) -> dict[str, Any]:
    """
    Ridge regression with permutation testing.

    Computes β = (X'X + λI)^{-1} X' Y with permutation-based significance testing.

    Parameters
    ----------
    X : ndarray, shape (n_genes, n_features)
        Design matrix (e.g., signature matrix).
        Rows are genes/observations, columns are features/proteins.
    Y : ndarray, shape (n_genes, n_samples)
        Response matrix (e.g., expression data).
        Rows are genes/observations, columns are samples.
    lambda_ : float, default=5e5
        Ridge regularization parameter (λ >= 0).
    n_rand : int, default=1000
        Number of permutations for significance testing.
        If 0, performs analytical t-test instead.
    seed : int, default=0
        Random seed for permutations. Use 0 for RidgeR compatibility.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Computation backend.
        - "auto": Use CuPy if available, else NumPy
        - "numpy": Force CPU computation
        - "cupy": Force GPU computation (raises error if unavailable)
    use_gsl_rng : bool, default=True
        Use GSL-compatible RNG for exact R/RidgeR reproducibility.
        Set to False for faster inference (~70x faster) when R matching is not needed.
    use_cache : bool, default=False
        Cache permutation tables to disk for reuse. Enable when running
        multiple analyses with the same gene count.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    dict
        Results dictionary containing:
        - beta : ndarray (n_features, n_samples) - Regression coefficients
        - se : ndarray (n_features, n_samples) - Standard errors
        - zscore : ndarray (n_features, n_samples) - Z-scores (or t-statistics if n_rand=0)
        - pvalue : ndarray (n_features, n_samples) - P-values
        - method : str - Backend used ("numpy" or "cupy")
        - time : float - Execution time in seconds
        - df : float (only if n_rand=0) - Degrees of freedom for t-test

    Examples
    --------
    >>> import numpy as np
    >>> from secactpy.ridge import ridge
    >>>
    >>> # Create test data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)  # 100 genes, 10 proteins
    >>> Y = np.random.randn(100, 5)   # 100 genes, 5 samples
    >>>
    >>> # Run ridge regression with permutation testing
    >>> result = ridge(X, Y, lambda_=5e5, n_rand=1000, seed=0)
    >>>
    >>> # Check significant results
    >>> significant = result['pvalue'] < 0.05
    >>> print(f"Significant coefficients: {significant.sum()}")

    Notes
    -----
    Results are identical to RidgeR when using:
    - Same seed (default: 0)
    - Same lambda value
    - Same number of permutations

    The algorithm uses Cholesky decomposition for numerical stability,
    matching RidgeR's GSL-based implementation.
    """
    start_time = time.time()

    # --- Input Validation ---
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got {Y.ndim}D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must have same number of rows (genes): "
            f"X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}")
    if n_rand < 0:
        raise ValueError(f"n_rand must be >= 0, got {n_rand}")

    n_genes, n_features = X.shape
    n_samples = Y.shape[1]

    if verbose:
        print(f"Ridge regression: {n_genes} genes, {n_features} features, {n_samples} samples")
        print(f"  lambda={lambda_}, n_rand={n_rand}, seed={seed}")

    # --- Backend Selection ---
    if backend == "auto":
        backend = "cupy" if CUPY_AVAILABLE else "numpy"
    elif backend == "cupy" and not CUPY_AVAILABLE:
        error_msg = "CuPy backend requested but not available."
        if CUPY_INIT_ERROR:
            error_msg += f" GPU initialization failed: {CUPY_INIT_ERROR}"
        else:
            error_msg += " Install CuPy with: pip install cupy-cuda11x (or cupy-cuda12x)"
        raise ImportError(error_msg)

    if verbose:
        print(f"  backend={backend}")

    # --- Dispatch to Backend ---
    if backend == "cupy":
        result = _ridge_cupy(X, Y, lambda_, n_rand, seed, use_gsl_rng, use_cache, verbose)
    else:
        if n_rand == 0:
            result = _ridge_ttest_numpy(X, Y, lambda_, verbose)
        else:
            result = _ridge_permutation_numpy(X, Y, lambda_, n_rand, seed, use_gsl_rng, use_cache, verbose)

    # --- Add Metadata ---
    result['method'] = backend
    result['time'] = time.time() - start_time

    if verbose:
        print(f"  completed in {result['time']:.3f}s")

    return result


# =============================================================================
# NumPy Backend - Permutation Test
# =============================================================================

def _ridge_permutation_numpy(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_: float,
    n_rand: int,
    seed: int,
    use_gsl_rng: bool,
    use_cache: bool,
    verbose: bool
) -> dict[str, np.ndarray]:
    """
    NumPy implementation of ridge regression with permutation testing.

    Uses T-column permutation which is mathematically equivalent to Y-row
    permutation but more efficient for GPU and sparse matrices:

        T[:, inv_perm] @ Y == T @ Y[perm, :]

    This produces identical results to RidgeR's Y-row permutation approach.
    """
    n_genes, n_features = X.shape
    n_samples = Y.shape[1]

    # --- Step 1: Compute T = (X'X + λI)^{-1} X' ---
    if verbose:
        print("  computing projection matrix T...")

    XtX = X.T @ X  # (n_features, n_features)
    XtX_reg = XtX + lambda_ * np.eye(n_features, dtype=np.float64)

    # Cholesky decomposition (matches GSL)
    try:
        L = linalg.cholesky(XtX_reg, lower=True)
        XtX_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
    except linalg.LinAlgError:
        # Fallback to pseudo-inverse if Cholesky fails
        warnings.warn("Cholesky decomposition failed, using pseudo-inverse")
        XtX_inv = linalg.pinv(XtX_reg)

    T = XtX_inv @ X.T  # (n_features, n_genes)

    # --- Step 2: Compute observed beta ---
    if verbose:
        print("  computing beta...")

    beta = T @ Y  # (n_features, n_samples)

    # --- Step 3: Permutation testing with T-column permutation ---
    if verbose:
        print(f"  running {n_rand} permutations (T-column method)...")

    # Generate inverse permutation table for T-column permutation
    # T[:, inv_perm] @ Y == T @ Y[perm, :] (mathematically equivalent)
    if use_gsl_rng:
        if use_cache:
            inv_perm_table = get_cached_inverse_perm_table(n_genes, n_rand, seed, verbose=verbose)
        else:
            # Generate directly without caching
            rng = GSLRNG(seed)
            inv_perm_table = rng.inverse_permutation_table(n_genes, n_rand)
    else:
        # Fast NumPy RNG (~70x faster, no caching needed)
        if verbose:
            print("  Generating permutation table (fast NumPy RNG)...")
        inv_perm_table = generate_inverse_permutation_table_fast(n_genes, n_rand, seed)

    # Accumulators
    aver = np.zeros((n_features, n_samples), dtype=np.float64)
    aver_sq = np.zeros((n_features, n_samples), dtype=np.float64)
    pvalue_counts = np.zeros((n_features, n_samples), dtype=np.float64)
    abs_beta = np.abs(beta)

    # Ensure T is contiguous for efficient column indexing
    T = np.ascontiguousarray(T)

    # Permutation loop with T-column permutation
    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]

        # Permute columns of T (equivalent to permuting rows of Y)
        T_perm = T[:, inv_perm_idx]

        # Compute permuted beta (Y stays in place)
        beta_perm = T_perm @ Y

        # Accumulate statistics
        pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

    # --- Step 4: Finalize statistics (matching RidgeR exactly) ---
    if verbose:
        print("  finalizing statistics...")

    # Variance of permutation distribution (for SE calculation)
    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)

    # Standard error (with protection for negative variance due to floating point)
    se = np.sqrt(np.maximum(var, 0.0))

    # Z-score: (beta - mean) / se
    # Protect against division by zero
    zscore = np.where(se > EPS, (beta - mean) / se, 0.0)

    # P-value: (count + 1) / (n_rand + 1)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)

    return {
        'beta': beta,
        'se': se,
        'zscore': zscore,
        'pvalue': pvalue
    }


# =============================================================================
# NumPy Backend - T-Test (n_rand=0)
# =============================================================================

def _ridge_ttest_numpy(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_: float,
    verbose: bool
) -> dict[str, np.ndarray]:
    """
    NumPy implementation of ridge regression with analytical t-test.

    Used when n_rand=0 for faster computation with parametric inference.
    """
    n_genes, n_features = X.shape

    # --- Step 1: Compute T = (X'X + λI)^{-1} X' ---
    if verbose:
        print("  computing projection matrix T...")

    XtX = X.T @ X
    XtX_reg = XtX + lambda_ * np.eye(n_features, dtype=np.float64)

    try:
        L = linalg.cholesky(XtX_reg, lower=True)
        XtX_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
    except linalg.LinAlgError:
        warnings.warn("Cholesky decomposition failed, using pseudo-inverse")
        XtX_inv = linalg.pinv(XtX_reg)

    T = XtX_inv @ X.T

    # --- Step 2: Compute beta ---
    if verbose:
        print("  computing beta...")

    beta = T @ Y

    # --- Step 3: Compute residuals and variance ---
    if verbose:
        print("  computing t-statistics...")

    # Predicted values
    Y_hat = X @ beta

    # Residuals
    residuals = Y - Y_hat

    # Residual sum of squares per sample
    rss = np.sum(residuals ** 2, axis=0)  # (n_samples,)

    # Degrees of freedom
    df = n_genes - n_features
    if df <= 0:
        warnings.warn(
            f"Degrees of freedom <= 0 ({df}). "
            "Results may be unreliable. Consider using permutation test."
        )
        df = max(df, 1)  # Prevent division by zero

    # Residual variance per sample
    sigma2 = rss / df  # (n_samples,)

    # Standard errors
    # SE_ij = sqrt(XtX_inv[i,i] * sigma2[j])
    var_beta_diag = np.diag(XtX_inv)  # (n_features,)
    se = np.sqrt(np.outer(var_beta_diag, sigma2))  # (n_features, n_samples)

    # --- Step 4: T-statistics and p-values ---
    # T-statistic
    zscore = np.where(se > EPS, beta / se, 0.0)

    # Two-sided p-value
    pvalue = 2.0 * stats.t.sf(np.abs(zscore), df=df)
    pvalue = np.clip(pvalue, 0.0, 1.0)

    return {
        'beta': beta,
        'se': se,
        'zscore': zscore,
        'pvalue': pvalue,
        'df': float(df)
    }


# =============================================================================
# CuPy Backend
# =============================================================================

def _ridge_cupy(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_: float,
    n_rand: int,
    seed: int,
    use_gsl_rng: bool,
    use_cache: bool,
    verbose: bool
) -> dict[str, np.ndarray]:
    """
    CuPy GPU implementation of ridge regression with permutation testing.

    Uses T-column permutation for better GPU efficiency:
        T[:, inv_perm] @ Y == T @ Y[perm, :]

    Y stays in place on GPU, only T column indices are shuffled.
    """
    if not CUPY_AVAILABLE or cp is None:
        raise RuntimeError("CuPy not available")

    if n_rand == 0:
        raise NotImplementedError(
            "T-test (n_rand=0) not implemented for CuPy backend. "
            "Use backend='numpy' for t-test."
        )

    n_genes, n_features = X.shape
    n_samples = Y.shape[1]

    # --- Transfer to GPU ---
    if verbose:
        print("  transferring data to GPU...")

    X_gpu = cp.asarray(X, dtype=cp.float64)
    Y_gpu = cp.asarray(Y, dtype=cp.float64)

    # --- Step 1: Compute T = (X'X + λI)^{-1} X' on GPU ---
    if verbose:
        print("  computing projection matrix T on GPU...")

    XtX = X_gpu.T @ X_gpu
    XtX_reg = XtX + lambda_ * cp.eye(n_features, dtype=cp.float64)

    # Cholesky decomposition on GPU
    try:
        L = cp.linalg.cholesky(XtX_reg)
        # Solve L @ L.T @ XtX_inv = I
        # First solve L @ Z = I
        I_gpu = cp.eye(n_features, dtype=cp.float64)
        Z = cp.linalg.solve(L, I_gpu)
        # Then solve L.T @ XtX_inv = Z
        XtX_inv = cp.linalg.solve(L.T, Z)
    except cp.linalg.LinAlgError:
        warnings.warn("GPU Cholesky failed, using pseudo-inverse")
        XtX_inv = cp.linalg.pinv(XtX_reg)

    T_gpu = XtX_inv @ X_gpu.T

    # Free intermediate GPU memory
    del XtX, XtX_reg, X_gpu
    if 'L' in dir():
        del L
    if 'Z' in dir():
        del Z
    del XtX_inv
    cp.get_default_memory_pool().free_all_blocks()

    # --- Step 2: Compute observed beta ---
    if verbose:
        print("  computing beta on GPU...")

    beta_gpu = T_gpu @ Y_gpu

    # --- Step 3: Permutation testing on GPU with T-column permutation ---
    if verbose:
        print(f"  running {n_rand} permutations on GPU (T-column method)...")

    # Generate inverse permutation table on CPU
    # T[:, inv_perm] @ Y == T @ Y[perm, :] (mathematically equivalent)
    if use_gsl_rng:
        if use_cache:
            inv_perm_table = get_cached_inverse_perm_table(n_genes, n_rand, seed, verbose=verbose)
        else:
            # Generate directly without caching
            rng = GSLRNG(seed)
            inv_perm_table = rng.inverse_permutation_table(n_genes, n_rand)
    else:
        # Fast NumPy RNG (~70x faster, no caching needed)
        if verbose:
            print("  Generating permutation table (fast NumPy RNG)...")
        inv_perm_table = generate_inverse_permutation_table_fast(n_genes, n_rand, seed)

    # Accumulators on GPU
    aver = cp.zeros((n_features, n_samples), dtype=cp.float64)
    aver_sq = cp.zeros((n_features, n_samples), dtype=cp.float64)
    pvalue_counts = cp.zeros((n_features, n_samples), dtype=cp.float64)
    abs_beta = cp.abs(beta_gpu)

    # Process permutations in batches to manage memory
    batch_size = min(100, n_rand)

    for batch_start in range(0, n_rand, batch_size):
        batch_end = min(batch_start + batch_size, n_rand)

        for i in range(batch_start, batch_end):
            inv_perm_idx = inv_perm_table[i]
            inv_perm_idx_gpu = cp.asarray(inv_perm_idx, dtype=cp.intp)

            # Permute columns of T (Y stays in place on GPU)
            T_perm = T_gpu[:, inv_perm_idx_gpu]

            # Compute permuted beta
            beta_perm = T_perm @ Y_gpu

            # Accumulate statistics
            pvalue_counts += (cp.abs(beta_perm) >= abs_beta).astype(cp.float64)
            aver += beta_perm
            aver_sq += beta_perm ** 2

            del inv_perm_idx_gpu, T_perm, beta_perm

        # Free memory periodically
        cp.get_default_memory_pool().free_all_blocks()

    # --- Step 4: Finalize statistics on GPU ---
    if verbose:
        print("  finalizing statistics on GPU...")

    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se_gpu = cp.sqrt(cp.maximum(var, 0.0))
    # Z-score: (beta - mean) / se
    zscore_gpu = cp.where(se_gpu > EPS, (beta_gpu - mean) / se_gpu, 0.0)
    pvalue_gpu = (pvalue_counts + 1.0) / (n_rand + 1.0)

    # --- Transfer results back to CPU ---
    if verbose:
        print("  transferring results to CPU...")

    beta = cp.asnumpy(beta_gpu)
    se = cp.asnumpy(se_gpu)
    zscore = cp.asnumpy(zscore_gpu)
    pvalue = cp.asnumpy(pvalue_gpu)

    # Cleanup GPU memory
    del T_gpu, Y_gpu, beta_gpu, aver, aver_sq, pvalue_counts
    del abs_beta, mean, var, se_gpu, zscore_gpu, pvalue_gpu
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return {
        'beta': beta,
        'se': se,
        'zscore': zscore,
        'pvalue': pvalue
    }


# =============================================================================
# Utility Functions
# =============================================================================

def compute_projection_matrix(
    X: np.ndarray,
    lambda_: float = DEFAULT_LAMBDA
) -> np.ndarray:
    """
    Compute the ridge regression projection matrix T = (X'X + λI)^{-1} X'.

    This can be precomputed when running multiple regressions with the
    same X matrix (e.g., in batch processing).

    Parameters
    ----------
    X : ndarray, shape (n_genes, n_features)
        Design matrix.
    lambda_ : float, default=5e5
        Ridge regularization parameter.

    Returns
    -------
    ndarray, shape (n_features, n_genes)
        Projection matrix T.
    """
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]

    XtX = X.T @ X
    XtX_reg = XtX + lambda_ * np.eye(n_features, dtype=np.float64)

    try:
        L = linalg.cholesky(XtX_reg, lower=True)
        XtX_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
    except linalg.LinAlgError:
        XtX_inv = linalg.pinv(XtX_reg)

    return XtX_inv @ X.T


def ridge_with_precomputed_T(
    T: np.ndarray,
    Y: np.ndarray,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    use_gsl_rng: bool = True
) -> dict[str, np.ndarray]:
    """
    Ridge regression using precomputed projection matrix.

    Uses T-column permutation for efficient computation:
        T[:, inv_perm] @ Y == T @ Y[perm, :]

    Useful for batch processing where the same X is used for multiple Y.

    Parameters
    ----------
    T : ndarray, shape (n_features, n_genes)
        Precomputed projection matrix from compute_projection_matrix().
    Y : ndarray, shape (n_genes, n_samples)
        Response matrix.
    n_rand : int, default=1000
        Number of permutations.
    seed : int, default=0
        Random seed.
    use_gsl_rng : bool, default=True
        Use GSL-compatible RNG for exact R/RidgeR reproducibility.
        Set to False for faster inference (~70x faster) when R matching is not needed.

    Returns
    -------
    dict
        Results dictionary (same as ridge()).
    """
    Y = np.asarray(Y, dtype=np.float64)
    T = np.ascontiguousarray(T)
    n_features, n_genes = T.shape
    n_samples = Y.shape[1]

    if Y.shape[0] != n_genes:
        raise ValueError(
            f"Y rows ({Y.shape[0]}) must match T columns ({n_genes})"
        )

    # Compute beta
    beta = T @ Y

    if n_rand == 0:
        raise NotImplementedError(
            "T-test requires X matrix. Use ridge() directly."
        )

    # Permutation testing with T-column permutation
    if use_gsl_rng:
        # GSL-compatible RNG for exact R/RidgeR reproducibility (uses caching)
        inv_perm_table = get_cached_inverse_perm_table(n_genes, n_rand, seed, verbose=False)
    else:
        # Fast NumPy RNG (~70x faster, no caching needed)
        inv_perm_table = generate_inverse_permutation_table_fast(n_genes, n_rand, seed)

    aver = np.zeros((n_features, n_samples), dtype=np.float64)
    aver_sq = np.zeros((n_features, n_samples), dtype=np.float64)
    pvalue_counts = np.zeros((n_features, n_samples), dtype=np.float64)
    abs_beta = np.abs(beta)

    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]
        T_perm = T[:, inv_perm_idx]
        beta_perm = T_perm @ Y

        pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se = np.sqrt(np.maximum(var, 0.0))
    # Z-score: (beta - mean) / se
    zscore = np.where(se > EPS, (beta - mean) / se, 0.0)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)

    return {
        'beta': beta,
        'se': se,
        'zscore': zscore,
        'pvalue': pvalue
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("SecActPy Ridge Module - Testing")
    print("=" * 60)

    # Set random seed for reproducible test data
    np.random.seed(42)

    # Create test data
    n_genes = 100
    n_features = 10
    n_samples = 5

    X = np.random.randn(n_genes, n_features)
    Y = np.random.randn(n_genes, n_samples)

    print(f"\nTest data: X({n_genes}, {n_features}), Y({n_genes}, {n_samples})")

    # Test 1: Basic permutation test
    print("\n1. Testing permutation test (NumPy)...")
    result = ridge(X, Y, lambda_=5e5, n_rand=100, seed=0, backend='numpy', verbose=True)
    print(f"   beta shape: {result['beta'].shape}")
    print(f"   pvalue range: [{result['pvalue'].min():.4f}, {result['pvalue'].max():.4f}]")
    print(f"   zscore range: [{result['zscore'].min():.4f}, {result['zscore'].max():.4f}]")

    # Test 2: T-test
    print("\n2. Testing t-test (n_rand=0)...")
    result_ttest = ridge(X, Y, lambda_=5e5, n_rand=0, backend='numpy', verbose=True)
    print(f"   df: {result_ttest['df']}")
    print(f"   pvalue range: [{result_ttest['pvalue'].min():.6f}, {result_ttest['pvalue'].max():.6f}]")

    # Test 3: Reproducibility
    print("\n3. Testing reproducibility...")
    result1 = ridge(X, Y, lambda_=5e5, n_rand=100, seed=0, backend='numpy')
    result2 = ridge(X, Y, lambda_=5e5, n_rand=100, seed=0, backend='numpy')

    if np.allclose(result1['beta'], result2['beta']) and \
       np.allclose(result1['pvalue'], result2['pvalue']):
        print("   ✓ Results are reproducible with same seed")
    else:
        print("   ✗ Results differ!")

    # Test 4: Different seeds
    print("\n4. Testing different seeds produce different results...")
    result_seed0 = ridge(X, Y, lambda_=5e5, n_rand=100, seed=0, backend='numpy')
    result_seed1 = ridge(X, Y, lambda_=5e5, n_rand=100, seed=1, backend='numpy')

    if not np.allclose(result_seed0['pvalue'], result_seed1['pvalue']):
        print("   ✓ Different seeds produce different p-values")
    else:
        print("   ✗ Seeds don't affect results!")

    # Test 5: Precomputed T
    print("\n5. Testing precomputed projection matrix...")
    T = compute_projection_matrix(X, lambda_=5e5)
    result_precomp = ridge_with_precomputed_T(T, Y, n_rand=100, seed=0)

    if np.allclose(result1['beta'], result_precomp['beta']) and \
       np.allclose(result1['pvalue'], result_precomp['pvalue']):
        print("   ✓ Precomputed T gives same results")
    else:
        print("   ✗ Precomputed T differs!")

    # Test 6: CuPy backend (if available)
    print(f"\n6. CuPy backend available: {CUPY_AVAILABLE}")
    if CUPY_AVAILABLE:
        print("   Testing CuPy backend...")
        result_gpu = ridge(X, Y, lambda_=5e5, n_rand=100, seed=0, backend='cupy', verbose=True)

        # Compare with NumPy results
        if np.allclose(result1['beta'], result_gpu['beta'], rtol=1e-10) and \
           np.allclose(result1['pvalue'], result_gpu['pvalue'], rtol=1e-10):
            print("   ✓ CuPy produces identical results to NumPy")
        else:
            beta_diff = np.abs(result1['beta'] - result_gpu['beta']).max()
            pval_diff = np.abs(result1['pvalue'] - result_gpu['pvalue']).max()
            print(f"   ✗ Results differ! max beta diff: {beta_diff:.2e}, max pvalue diff: {pval_diff:.2e}")

    # Test 7: Performance benchmark
    print("\n7. Performance benchmark...")
    n_genes_bench = 1000
    n_features_bench = 50
    n_samples_bench = 10
    n_rand_bench = 100

    X_bench = np.random.randn(n_genes_bench, n_features_bench)
    Y_bench = np.random.randn(n_genes_bench, n_samples_bench)

    print(f"   Data: {n_genes_bench} genes, {n_features_bench} features, {n_samples_bench} samples")
    print(f"   Permutations: {n_rand_bench}")

    start = time.time()
    _ = ridge(X_bench, Y_bench, n_rand=n_rand_bench, backend='numpy')
    numpy_time = time.time() - start
    print(f"   NumPy: {numpy_time:.3f}s")

    if CUPY_AVAILABLE:
        # Warmup
        _ = ridge(X_bench, Y_bench, n_rand=10, backend='cupy')

        start = time.time()
        _ = ridge(X_bench, Y_bench, n_rand=n_rand_bench, backend='cupy')
        cupy_time = time.time() - start
        print(f"   CuPy:  {cupy_time:.3f}s (speedup: {numpy_time/cupy_time:.1f}x)")

    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
