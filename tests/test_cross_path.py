"""
Cross-path consistency tests: verify that all code paths
(cpu dense/sparse, standard/batch) produce equivalent results
and match R SecAct output.

Paths tested:
    1. ridge()       + dense Y (pre-scaled)          → R reference
    2. ridge_batch() + dense Y (pre-scaled)          → R reference
    3. ridge()       + sparse Y + col normalization   → R reference
    4. ridge_batch() + sparse Y + col normalization   → R reference
    5. ridge_batch() + sparse Y + sparse_mode=True    → R reference
    6. ridge()       + sparse Y + sparse_mode=True    → R reference

All paths use the same bulk RNA-seq dataset and should match the R
SecAct.inference.gsl.legacy reference to within 1e-8 tolerance.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sps
import pytest
from pathlib import Path

from secactpy import load_signature
from secactpy.ridge import ridge
from secactpy.batch import ridge_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
INPUT_FILE = DATA_DIR / "input" / "Ly86-Fc_vs_Vehicle_logFC.txt"
R_OUTPUT_DIR = DATA_DIR / "output" / "ridge" / "bulk"

# Ridge parameters matching R
LAMBDA = 5e5
NRAND = 1000
SEED = 0
TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_r_reference():
    """Load R SecAct reference output."""
    result = {}
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        filepath = R_OUTPUT_DIR / f"{name}.txt"
        if filepath.exists():
            df = pd.read_csv(filepath, sep=r'\s+', index_col=0)
            result[name] = df
    return result


def _prepare_data():
    """Prepare input data matching test_ridge.py's secact_inference_gsl_legacy."""
    sig_df = load_signature('secact')
    Y_df = pd.read_csv(INPUT_FILE, sep=r'\s+', index_col=0)

    common_genes = Y_df.index.intersection(sig_df.index)
    X_aligned = sig_df.loc[common_genes].astype(np.float64)
    Y_aligned = Y_df.loc[common_genes].astype(np.float64)

    # Scale (R's scale() uses ddof=1)
    X_scaled = (X_aligned - X_aligned.mean()) / X_aligned.std(ddof=1)
    Y_scaled = (Y_aligned - Y_aligned.mean()) / Y_aligned.std(ddof=1)
    X_scaled = X_scaled.fillna(0)
    Y_scaled = Y_scaled.fillna(0)

    return X_scaled, Y_scaled, common_genes


def _assert_matches_r(result, r_ref, feature_names, sample_names, label):
    """Assert that a result dict matches R reference within tolerance."""
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        py_df = pd.DataFrame(result[name], index=feature_names,
                              columns=sample_names)
        r_df = r_ref[name]
        py_aligned = py_df.loc[r_df.index, r_df.columns]
        diff = np.abs(py_aligned.values - r_df.values)
        max_diff = np.nanmax(diff)
        assert max_diff < TOLERANCE, (
            f"[{label}] {name} max diff = {max_diff:.2e} >= {TOLERANCE}"
        )


def _assert_results_match(r1, r2, label, tol=1e-10):
    """Assert two result dicts match within tolerance."""
    for key in ['beta', 'se', 'zscore', 'pvalue']:
        diff = np.abs(r1[key] - r2[key]).max()
        assert diff < tol, f"[{label}] {key} max diff = {diff:.2e} >= {tol}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bulk_data():
    """Load and prepare bulk data once per module."""
    if not INPUT_FILE.exists() or not R_OUTPUT_DIR.exists():
        pytest.skip("Bulk dataset or R reference not available")
    X_scaled, Y_scaled, common_genes = _prepare_data()
    r_ref = _load_r_reference()
    if not r_ref:
        pytest.skip("R reference files not found")
    return X_scaled, Y_scaled, common_genes, r_ref


# ---------------------------------------------------------------------------
# 1. Dense ridge() vs R
# ---------------------------------------------------------------------------

def test_dense_ridge_vs_r(bulk_data):
    """Dense ridge() should match R SecAct output exactly."""
    X_scaled, Y_scaled, common_genes, r_ref = bulk_data
    X = X_scaled.values
    Y = Y_scaled.values
    result = ridge(X, Y, lambda_=LAMBDA, n_rand=NRAND, seed=SEED,
                   backend='numpy', use_cache=True)
    _assert_matches_r(result, r_ref,
                      X_scaled.columns.tolist(),
                      Y_scaled.columns.tolist(),
                      "dense ridge()")


# ---------------------------------------------------------------------------
# 2. Dense ridge_batch() vs R
# ---------------------------------------------------------------------------

def test_dense_batch_vs_r(bulk_data):
    """Dense ridge_batch() should match R SecAct output exactly."""
    X_scaled, Y_scaled, common_genes, r_ref = bulk_data
    X = X_scaled.values
    Y = Y_scaled.values
    result = ridge_batch(X, Y, lambda_=LAMBDA, n_rand=NRAND, seed=SEED,
                         batch_size=100, backend='numpy', use_cache=True)
    _assert_matches_r(result, r_ref,
                      X_scaled.columns.tolist(),
                      Y_scaled.columns.tolist(),
                      "dense ridge_batch()")


# ---------------------------------------------------------------------------
# 3. Sparse ridge() + col normalization vs R
# ---------------------------------------------------------------------------

def test_sparse_ridge_col_norm_vs_r(bulk_data):
    """Sparse ridge(col_center=True, col_scale=True) should match R output.

    The sparse path applies in-flight z-scoring (center + scale), which
    should be equivalent to pre-scaled dense input.
    """
    X_scaled, Y_scaled, common_genes, r_ref = bulk_data
    X = X_scaled.values
    # Raw (unscaled) Y as sparse — the col_center/col_scale flags do the work
    Y_raw = pd.read_csv(INPUT_FILE, sep=r'\s+', index_col=0)
    Y_raw = Y_raw.loc[common_genes].astype(np.float64).fillna(0)
    Y_sparse = sps.csc_matrix(Y_raw.values)

    result = ridge(X, Y_sparse, lambda_=LAMBDA, n_rand=NRAND, seed=SEED,
                   backend='numpy', sparse_mode=True,
                   col_center=True, col_scale=True, use_cache=True)

    _assert_matches_r(result, r_ref,
                      X_scaled.columns.tolist(),
                      Y_scaled.columns.tolist(),
                      "sparse ridge(col_center/col_scale)")


# ---------------------------------------------------------------------------
# 4. Sparse ridge_batch() + col normalization vs R
# ---------------------------------------------------------------------------

def test_sparse_batch_col_norm_vs_r(bulk_data):
    """Sparse ridge_batch(col_center=True, col_scale=True) should match R."""
    X_scaled, Y_scaled, common_genes, r_ref = bulk_data
    X = X_scaled.values
    Y_raw = pd.read_csv(INPUT_FILE, sep=r'\s+', index_col=0)
    Y_raw = Y_raw.loc[common_genes].astype(np.float64).fillna(0)
    Y_sparse = sps.csc_matrix(Y_raw.values)

    result = ridge_batch(X, Y_sparse, lambda_=LAMBDA, n_rand=NRAND, seed=SEED,
                         batch_size=100, backend='numpy',
                         col_center=True, col_scale=True, use_cache=True)

    _assert_matches_r(result, r_ref,
                      X_scaled.columns.tolist(),
                      Y_scaled.columns.tolist(),
                      "sparse ridge_batch(col_center/col_scale)")


# ---------------------------------------------------------------------------
# 5. Sparse ridge_batch() sparse_mode=True vs R
# ---------------------------------------------------------------------------

def test_sparse_batch_sparse_mode_col_norm_vs_r(bulk_data):
    """Sparse ridge_batch(sparse_mode=True, col_center/scale=True) vs R."""
    X_scaled, Y_scaled, common_genes, r_ref = bulk_data
    X = X_scaled.values
    Y_raw = pd.read_csv(INPUT_FILE, sep=r'\s+', index_col=0)
    Y_raw = Y_raw.loc[common_genes].astype(np.float64).fillna(0)
    Y_sparse = sps.csc_matrix(Y_raw.values)

    result = ridge_batch(X, Y_sparse, lambda_=LAMBDA, n_rand=NRAND, seed=SEED,
                         batch_size=100, backend='numpy', sparse_mode=True,
                         col_center=True, col_scale=True, use_cache=True)

    _assert_matches_r(result, r_ref,
                      X_scaled.columns.tolist(),
                      Y_scaled.columns.tolist(),
                      "sparse ridge_batch(sparse_mode=True, col_norm)")


# ---------------------------------------------------------------------------
# 6. Cross-path consistency: all 4 paths produce identical beta
# ---------------------------------------------------------------------------

def test_all_paths_identical_beta(bulk_data):
    """All code paths (dense/sparse × standard/batch) should produce the same beta."""
    X_scaled, Y_scaled, common_genes, r_ref = bulk_data
    X = X_scaled.values
    Y_dense = Y_scaled.values

    Y_raw = pd.read_csv(INPUT_FILE, sep=r'\s+', index_col=0)
    Y_raw = Y_raw.loc[common_genes].astype(np.float64).fillna(0)
    Y_sparse = sps.csc_matrix(Y_raw.values)

    kw = dict(lambda_=LAMBDA, n_rand=NRAND, seed=SEED, backend='numpy',
              use_cache=True)

    r_dense = ridge(X, Y_dense, **kw)
    r_batch_dense = ridge_batch(X, Y_dense, batch_size=100, **kw)
    r_sparse = ridge(X, Y_sparse, sparse_mode=True,
                     col_center=True, col_scale=True, **kw)
    r_batch_sparse = ridge_batch(X, Y_sparse, batch_size=100,
                                 col_center=True, col_scale=True, **kw)
    r_batch_sparse_mode = ridge_batch(X, Y_sparse, batch_size=100,
                                      sparse_mode=True,
                                      col_center=True, col_scale=True, **kw)

    _assert_results_match(r_dense, r_batch_dense,
                          "dense ridge vs dense batch")
    _assert_results_match(r_dense, r_sparse,
                          "dense ridge vs sparse ridge", tol=TOLERANCE)
    _assert_results_match(r_dense, r_batch_sparse,
                          "dense ridge vs sparse batch", tol=TOLERANCE)
    _assert_results_match(r_dense, r_batch_sparse_mode,
                          "dense ridge vs sparse batch(sparse_mode)", tol=TOLERANCE)
    _assert_results_match(r_sparse, r_batch_sparse,
                          "sparse ridge vs sparse batch", tol=TOLERANCE)
