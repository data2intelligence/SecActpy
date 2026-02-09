"""
Tests for col_center / col_scale parameters in ridge() and ridge_batch()
sparse paths.

Verifies that the four col_center × col_scale combinations produce
correct results, and that sparse_mode=True/False are consistent.
"""

import numpy as np
import scipy.sparse as sps
import pytest

from secactpy.ridge import ridge
from secactpy.batch import ridge_batch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data():
    """Small reproducible test dataset with sparse Y."""
    np.random.seed(42)
    n_genes, n_features, n_samples = 200, 10, 30
    X = np.random.randn(n_genes, n_features)
    Y_sparse = sps.random(n_genes, n_samples, density=0.05, format='csc',
                          random_state=42, dtype=np.float64)
    return X, Y_sparse


COMMON_KW = dict(lambda_=5e5, n_rand=50, seed=0, batch_size=100,
                 backend='numpy', rng_method='gsl')


# ---------------------------------------------------------------------------
# 1. Default (col_center=True, col_scale=True) matches old behaviour
# ---------------------------------------------------------------------------

def test_default_backward_compat(data):
    """col_center=True, col_scale=True should match the result without flags."""
    X, Y_sparse = data
    result_default = ridge_batch(X, Y_sparse, **COMMON_KW)
    result_explicit = ridge_batch(X, Y_sparse, col_center=True,
                                  col_scale=True, **COMMON_KW)
    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_explicit[key], result_default[key],
            atol=1e-12, rtol=1e-12,
            err_msg=f"backward compat mismatch for {key}"
        )


# ---------------------------------------------------------------------------
# 2. col_center=False, col_scale=False  →  raw T @ Y
# ---------------------------------------------------------------------------

def test_no_center_no_scale_matches_raw(data):
    """With both flags off, beta should equal raw T @ Y.toarray()."""
    X, Y_sparse = data
    result = ridge_batch(X, Y_sparse, col_center=False, col_scale=False,
                         **COMMON_KW)

    # Manually compute T
    XtX = X.T @ X
    n_features = X.shape[1]
    T = np.linalg.solve(XtX + 5e5 * np.eye(n_features), X.T)
    expected_beta = T @ Y_sparse.toarray()

    np.testing.assert_allclose(
        result['beta'], expected_beta,
        atol=1e-10, rtol=1e-10,
        err_msg="raw projection mismatch"
    )


# ---------------------------------------------------------------------------
# 3. col_center=True, col_scale=False  →  T @ (Y - col_mean)
# ---------------------------------------------------------------------------

def test_center_no_scale(data):
    """col_center=True, col_scale=False should match T @ (Y - mu)."""
    X, Y_sparse = data
    result = ridge_batch(X, Y_sparse, col_center=True, col_scale=False,
                         **COMMON_KW)

    Y_dense = Y_sparse.toarray()
    mu = Y_dense.mean(axis=0)
    Y_centered = Y_dense - mu

    XtX = X.T @ X
    n_features = X.shape[1]
    T = np.linalg.solve(XtX + 5e5 * np.eye(n_features), X.T)
    expected_beta = T @ Y_centered

    np.testing.assert_allclose(
        result['beta'], expected_beta,
        atol=1e-10, rtol=1e-10,
        err_msg="center-only mismatch"
    )


# ---------------------------------------------------------------------------
# 4. col_center=False, col_scale=True  →  (T @ Y) / sigma
# ---------------------------------------------------------------------------

def test_no_center_scale(data):
    """col_center=False, col_scale=True should match (T @ Y) / sigma."""
    X, Y_sparse = data
    result = ridge_batch(X, Y_sparse, col_center=False, col_scale=True,
                         **COMMON_KW)

    Y_dense = Y_sparse.toarray()
    sigma = Y_dense.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-15, 1.0, sigma)

    XtX = X.T @ X
    n_features = X.shape[1]
    T = np.linalg.solve(XtX + 5e5 * np.eye(n_features), X.T)
    expected_beta = (T @ Y_dense) / sigma

    np.testing.assert_allclose(
        result['beta'], expected_beta,
        atol=1e-10, rtol=1e-10,
        err_msg="scale-only mismatch"
    )


# ---------------------------------------------------------------------------
# 5. Combinations with row_center=True
# ---------------------------------------------------------------------------

def test_row_center_with_col_flags(data):
    """row_center + each col flag combo should be self-consistent.

    We verify col_center=False, col_scale=False, row_center=True gives
    T @ (Y - row_means).
    """
    X, Y_sparse = data
    result = ridge_batch(X, Y_sparse, row_center=True,
                         col_center=False, col_scale=False, **COMMON_KW)

    Y_dense = Y_sparse.toarray()
    row_means = Y_dense.mean(axis=1, keepdims=True)
    Y_row_centered = Y_dense - row_means

    XtX = X.T @ X
    n_features = X.shape[1]
    T = np.linalg.solve(XtX + 5e5 * np.eye(n_features), X.T)
    expected_beta = T @ Y_row_centered

    np.testing.assert_allclose(
        result['beta'], expected_beta,
        atol=1e-10, rtol=1e-10,
        err_msg="row_center + no col norm mismatch"
    )


def test_row_center_with_full_normalization(data):
    """row_center=True with default col_center/col_scale should match
    the full formula: (T @ (Y - row_mean)) / sigma' - c * (mu'/sigma')
    """
    X, Y_sparse = data

    result_default = ridge_batch(X, Y_sparse, row_center=True, **COMMON_KW)
    result_explicit = ridge_batch(X, Y_sparse, row_center=True,
                                  col_center=True, col_scale=True,
                                  **COMMON_KW)

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_explicit[key], result_default[key],
            atol=1e-12, rtol=1e-12,
            err_msg=f"row_center backward compat mismatch for {key}"
        )


# ---------------------------------------------------------------------------
# 6. sparse_mode=True vs False consistency for each flag combo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col_center,col_scale", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_sparse_mode_consistency(data, col_center, col_scale):
    """sparse_mode=True and False should give identical results."""
    X, Y_sparse = data
    kw = dict(col_center=col_center, col_scale=col_scale, **COMMON_KW)

    result_dense_path = ridge_batch(X, Y_sparse, sparse_mode=False, **kw)
    result_sparse_path = ridge_batch(X, Y_sparse, sparse_mode=True, **kw)

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse_path[key], result_dense_path[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"sparse_mode mismatch for {key} "
                    f"(col_center={col_center}, col_scale={col_scale})"
        )


@pytest.mark.parametrize("col_center,col_scale", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_sparse_mode_consistency_with_row_center(data, col_center, col_scale):
    """sparse_mode=True and False should agree with row_center=True."""
    X, Y_sparse = data
    kw = dict(col_center=col_center, col_scale=col_scale,
              row_center=True, **COMMON_KW)

    result_dense_path = ridge_batch(X, Y_sparse, sparse_mode=False, **kw)
    result_sparse_path = ridge_batch(X, Y_sparse, sparse_mode=True, **kw)

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse_path[key], result_dense_path[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"sparse_mode+row_center mismatch for {key} "
                    f"(col_center={col_center}, col_scale={col_scale})"
        )


# ===========================================================================
# ridge() (non-batch) col_center / col_scale tests
# ===========================================================================

RIDGE_KW = dict(lambda_=5e5, n_rand=50, seed=0, backend='numpy',
                rng_method='gsl')


# ---------------------------------------------------------------------------
# 7. ridge() default matches explicit True/True
# ---------------------------------------------------------------------------

def test_ridge_default_backward_compat(data):
    """ridge() col_center=True, col_scale=True should match default."""
    X, Y_sparse = data
    result_default = ridge(X, Y_sparse, sparse_mode=True, **RIDGE_KW)
    result_explicit = ridge(X, Y_sparse, sparse_mode=True,
                            col_center=True, col_scale=True, **RIDGE_KW)
    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_explicit[key], result_default[key],
            atol=1e-12, rtol=1e-12,
            err_msg=f"ridge backward compat mismatch for {key}"
        )


# ---------------------------------------------------------------------------
# 8. ridge() col_center=False, col_scale=False → raw T @ Y
# ---------------------------------------------------------------------------

def test_ridge_no_center_no_scale_matches_raw(data):
    """ridge(col_center=False, col_scale=False) should equal raw T @ Y."""
    X, Y_sparse = data
    result = ridge(X, Y_sparse, sparse_mode=True,
                   col_center=False, col_scale=False, **RIDGE_KW)

    XtX = X.T @ X
    n_features = X.shape[1]
    T = np.linalg.solve(XtX + 5e5 * np.eye(n_features), X.T)
    expected_beta = T @ Y_sparse.toarray()

    np.testing.assert_allclose(
        result['beta'], expected_beta,
        atol=1e-10, rtol=1e-10,
        err_msg="ridge raw projection mismatch"
    )


# ---------------------------------------------------------------------------
# 9. ridge() col_center=True, col_scale=False → T @ (Y - mu)
# ---------------------------------------------------------------------------

def test_ridge_center_no_scale(data):
    """ridge(col_center=True, col_scale=False) should match T @ (Y - mu)."""
    X, Y_sparse = data
    result = ridge(X, Y_sparse, sparse_mode=True,
                   col_center=True, col_scale=False, **RIDGE_KW)

    Y_dense = Y_sparse.toarray()
    mu = Y_dense.mean(axis=0)
    Y_centered = Y_dense - mu

    XtX = X.T @ X
    n_features = X.shape[1]
    T = np.linalg.solve(XtX + 5e5 * np.eye(n_features), X.T)
    expected_beta = T @ Y_centered

    np.testing.assert_allclose(
        result['beta'], expected_beta,
        atol=1e-10, rtol=1e-10,
        err_msg="ridge center-only mismatch"
    )


# ---------------------------------------------------------------------------
# 10. ridge() col_center=False, col_scale=True → (T @ Y) / sigma
# ---------------------------------------------------------------------------

def test_ridge_no_center_scale(data):
    """ridge(col_center=False, col_scale=True) should match (T @ Y) / sigma."""
    X, Y_sparse = data
    result = ridge(X, Y_sparse, sparse_mode=True,
                   col_center=False, col_scale=True, **RIDGE_KW)

    Y_dense = Y_sparse.toarray()
    sigma = Y_dense.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)

    XtX = X.T @ X
    n_features = X.shape[1]
    T = np.linalg.solve(XtX + 5e5 * np.eye(n_features), X.T)
    expected_beta = (T @ Y_dense) / sigma

    np.testing.assert_allclose(
        result['beta'], expected_beta,
        atol=1e-10, rtol=1e-10,
        err_msg="ridge scale-only mismatch"
    )


# ---------------------------------------------------------------------------
# 11. ridge() vs ridge_batch() consistency for each flag combo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col_center,col_scale", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_ridge_vs_batch_consistency(data, col_center, col_scale):
    """ridge() and ridge_batch() should produce identical beta for each combo."""
    X, Y_sparse = data
    result_ridge = ridge(X, Y_sparse, sparse_mode=True,
                         col_center=col_center, col_scale=col_scale,
                         **RIDGE_KW)
    result_batch = ridge_batch(X, Y_sparse, sparse_mode=True,
                               col_center=col_center, col_scale=col_scale,
                               **COMMON_KW)

    np.testing.assert_allclose(
        result_ridge['beta'], result_batch['beta'],
        atol=1e-10, rtol=1e-10,
        err_msg=f"ridge vs batch beta mismatch "
                f"(col_center={col_center}, col_scale={col_scale})"
    )
