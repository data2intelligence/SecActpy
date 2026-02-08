"""
Tests for sparse_mode parameter in ridge() and ridge_batch().

Verifies that sparse_mode=True produces results identical to the
dense path when Y is sparse.
"""

import numpy as np
import scipy.sparse as sps
import pytest


def test_ridge_sparse_mode_matches_dense():
    """ridge(sparse_mode=True) should match ridge(dense Y) exactly."""
    from secactpy.ridge import ridge

    np.random.seed(42)
    n_genes, n_features, n_samples = 200, 10, 20
    X = np.random.randn(n_genes, n_features)

    # Create sparse Y (2% density)
    Y_sparse = sps.random(n_genes, n_samples, density=0.02, format='csc',
                          random_state=42, dtype=np.float64)
    Y_dense = Y_sparse.toarray()

    result_dense = ridge(X, Y_dense, lambda_=5e5, n_rand=50, seed=0,
                         backend='numpy')
    result_sparse = ridge(X, Y_sparse, lambda_=5e5, n_rand=50, seed=0,
                          backend='numpy', sparse_mode=True)

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse[key], result_dense[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"sparse_mode mismatch for {key}"
        )


def test_ridge_sparse_mode_false_densifies():
    """ridge(sparse_mode=False) with sparse Y should densify and work normally."""
    from secactpy.ridge import ridge

    np.random.seed(42)
    n_genes, n_features, n_samples = 100, 5, 10
    X = np.random.randn(n_genes, n_features)
    Y_sparse = sps.random(n_genes, n_samples, density=0.05, format='csr',
                          random_state=42, dtype=np.float64)
    Y_dense = Y_sparse.toarray()

    result_default = ridge(X, Y_sparse, lambda_=5e5, n_rand=50, seed=0,
                           backend='numpy', sparse_mode=False)
    result_dense = ridge(X, Y_dense, lambda_=5e5, n_rand=50, seed=0,
                         backend='numpy')

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_default[key], result_dense[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"default sparse handling mismatch for {key}"
        )


def test_ridge_sparse_mode_ttest_fallback():
    """ridge(sparse_mode=True, n_rand=0) should fall back to dense t-test."""
    from secactpy.ridge import ridge

    np.random.seed(42)
    n_genes, n_features, n_samples = 100, 5, 10
    X = np.random.randn(n_genes, n_features)
    Y_sparse = sps.random(n_genes, n_samples, density=0.05, format='csc',
                          random_state=42, dtype=np.float64)
    Y_dense = Y_sparse.toarray()

    result_sparse = ridge(X, Y_sparse, lambda_=5e5, n_rand=0,
                          backend='numpy', sparse_mode=True)
    result_dense = ridge(X, Y_dense, lambda_=5e5, n_rand=0,
                         backend='numpy')

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse[key], result_dense[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"t-test fallback mismatch for {key}"
        )


def test_ridge_sparse_mode_csr_input():
    """ridge(sparse_mode=True) should handle CSR input (auto-converts to CSC)."""
    from secactpy.ridge import ridge

    np.random.seed(42)
    n_genes, n_features, n_samples = 100, 5, 10
    X = np.random.randn(n_genes, n_features)
    Y_csr = sps.random(n_genes, n_samples, density=0.02, format='csr',
                       random_state=42, dtype=np.float64)
    Y_dense = Y_csr.toarray()

    result_sparse = ridge(X, Y_csr, lambda_=5e5, n_rand=50, seed=0,
                          backend='numpy', sparse_mode=True)
    result_dense = ridge(X, Y_dense, lambda_=5e5, n_rand=50, seed=0,
                         backend='numpy')

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse[key], result_dense[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"CSR input mismatch for {key}"
        )


def test_ridge_batch_sparse_mode():
    """ridge_batch(sparse_mode=True) should match ridge_batch(sparse_mode=False)."""
    from secactpy.batch import ridge_batch

    np.random.seed(42)
    n_genes, n_features, n_samples = 200, 10, 50
    X = np.random.randn(n_genes, n_features)
    Y_sparse = sps.random(n_genes, n_samples, density=0.02, format='csc',
                          random_state=42, dtype=np.float64)

    result_dense_path = ridge_batch(
        X, Y_sparse, lambda_=5e5, n_rand=50, seed=0,
        batch_size=20, backend='numpy', sparse_mode=False
    )
    result_sparse_path = ridge_batch(
        X, Y_sparse, lambda_=5e5, n_rand=50, seed=0,
        batch_size=20, backend='numpy', sparse_mode=True
    )

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse_path[key], result_dense_path[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"batch sparse_mode mismatch for {key}"
        )


def test_ridge_dense_y_unaffected_by_sparse_mode():
    """sparse_mode=True with dense Y should behave identically to sparse_mode=False."""
    from secactpy.ridge import ridge

    np.random.seed(42)
    n_genes, n_features, n_samples = 100, 5, 10
    X = np.random.randn(n_genes, n_features)
    Y = np.random.randn(n_genes, n_samples)

    result_normal = ridge(X, Y, lambda_=5e5, n_rand=50, seed=0,
                          backend='numpy', sparse_mode=False)
    result_sparse_flag = ridge(X, Y, lambda_=5e5, n_rand=50, seed=0,
                               backend='numpy', sparse_mode=True)

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse_flag[key], result_normal[key],
            atol=1e-10, rtol=1e-10,
            err_msg=f"dense Y with sparse_mode=True mismatch for {key}"
        )


def test_ridge_batch_sparse_mode_with_row_center():
    """ridge_batch(sparse_mode=True, row_center=True) should match dense row-centered path."""
    from secactpy.batch import ridge_batch

    np.random.seed(42)
    n_genes, n_features, n_samples = 200, 10, 50
    X = np.random.randn(n_genes, n_features)

    # Create sparse Y (2% density)
    Y_sparse = sps.random(n_genes, n_samples, density=0.02, format='csc',
                          random_state=42, dtype=np.float64)
    Y_dense = Y_sparse.toarray()

    # Dense path: row-center Y, then column z-score, then ridge_batch
    row_means = Y_dense.mean(axis=1, keepdims=True)
    Y_centered = Y_dense - row_means
    col_mu = Y_centered.mean(axis=0)
    col_sigma = Y_centered.std(axis=0, ddof=1)
    col_sigma[col_sigma == 0] = 1.0
    Y_scaled = (Y_centered - col_mu) / col_sigma

    result_dense = ridge_batch(
        X, Y_scaled, lambda_=5e5, n_rand=50, seed=0,
        batch_size=20, backend='numpy'
    )

    # Sparse path: sparse_mode + row_center handles normalization in-flight
    result_sparse = ridge_batch(
        X, Y_sparse, lambda_=5e5, n_rand=50, seed=0,
        batch_size=20, backend='numpy', sparse_mode=True, row_center=True
    )

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse[key], result_dense[key],
            atol=1e-8, rtol=1e-8,
            err_msg=f"batch sparse+row_center mismatch for {key}"
        )


def test_end_to_end_sparse_cpm_log2_pipeline():
    """Full pipeline: sparse CPM → log2 → ridge with row_center matches dense equivalent."""
    from secactpy.batch import ridge_batch

    np.random.seed(42)
    n_genes, n_features, n_samples = 200, 10, 100

    # Signature matrix (z-scored)
    X = np.random.randn(n_genes, n_features)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    # Simulate sparse count matrix (like scRNA-seq)
    Y_sparse = sps.random(n_genes, n_samples, density=0.03, format='csc',
                          random_state=42, dtype=np.float64)
    # Scale to look like counts
    Y_sparse = (Y_sparse * 100).astype(np.float64)

    # === Dense path: CPM → log2 → row center → col z-score → ridge ===
    Y_dense = Y_sparse.toarray()
    col_sums = Y_dense.sum(axis=0)
    Y_cpm = Y_dense / col_sums * 1e5
    Y_log = np.log2(Y_cpm + 1)
    row_means = Y_log.mean(axis=1, keepdims=True)
    Y_centered = Y_log - row_means
    col_mu = Y_centered.mean(axis=0)
    col_sigma = Y_centered.std(axis=0, ddof=1)
    col_sigma[col_sigma == 0] = 1.0
    Y_scaled = (Y_centered - col_mu) / col_sigma

    result_dense = ridge_batch(
        X, Y_scaled, lambda_=5e5, n_rand=50, seed=0,
        batch_size=40, backend='numpy'
    )

    # === Sparse path: CPM + log2 on sparse → ridge_batch(sparse_mode, row_center) ===
    from scipy.sparse import diags
    col_sums_sp = np.asarray(Y_sparse.sum(axis=0)).ravel()
    scaling = diags(1e5 / col_sums_sp)
    Y_sp_cpm = Y_sparse.astype(np.float64) @ scaling
    Y_sp_cpm = Y_sp_cpm.tocsc()
    Y_sp_cpm.data = np.log2(Y_sp_cpm.data + 1)

    result_sparse = ridge_batch(
        X, Y_sp_cpm, lambda_=5e5, n_rand=50, seed=0,
        batch_size=40, backend='numpy', sparse_mode=True, row_center=True
    )

    for key in ['beta', 'se', 'zscore', 'pvalue']:
        np.testing.assert_allclose(
            result_sparse[key], result_dense[key],
            atol=1e-8, rtol=1e-8,
            err_msg=f"end-to-end pipeline mismatch for {key}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
