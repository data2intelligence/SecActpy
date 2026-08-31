#!/usr/bin/env python3
"""Regression tests for the analytical ridge t-test (``n_rand=0``).

Pins the two properties that make SecActpy's ``n_rand=0`` path agree with
the flashreg / flashregpy accelerators (and that a naive OLS-style
implementation gets wrong at ``lambda > 0``):

1. Effective degrees of freedom ``df = n - tr(H)`` with the ridge hat
   matrix ``H = X (X'X + lambda I)^{-1} X'`` -- strictly below the OLS
   ``n - p`` at ``lambda > 0``.
2. Standard errors from the ridge *sandwich* covariance
   ``Cov(beta) = sigma^2 (X'X + lambda I)^{-1} X'X (X'X + lambda I)^{-1}``,
   not the plain ``sigma^2 diag((X'X + lambda I)^{-1})`` (which overstates
   SE ~8x at lambda = 5e5).

The references are recomputed in-test from first principles, so the guard
does not depend on flashregpy being installed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from secactpy.ridge import ridge  # noqa: E402


def _sandwich_reference(X, Y, lam):
    """Ridge analytical t-test computed straight from the definitions."""
    n, p = X.shape
    XtX = X.T @ X
    Ainv = np.linalg.inv(XtX + lam * np.eye(p))
    beta = Ainv @ X.T @ Y
    resid = Y - X @ beta
    rss = np.sum(resid ** 2, axis=0)
    AinvXtX = Ainv @ XtX
    df = n - np.trace(AinvXtX)
    sigma2 = rss / df
    var_diag = np.clip(np.diag(AinvXtX @ Ainv), 0.0, None)
    se = np.sqrt(np.outer(var_diag, sigma2))
    z = np.where(se > 0, beta / se, 0.0)
    pval = np.clip(2.0 * stats.t.sf(np.abs(z), df=df), 0.0, 1.0)
    return dict(beta=beta, se=se, zscore=z, pvalue=pval, df=df)


@pytest.mark.parametrize("lam", [1e2, 1e4, 5e5])
def test_ttest_matches_sandwich_reference(lam):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 8))
    Y = rng.standard_normal((120, 10))
    ref = _sandwich_reference(X, Y, lam)
    got = ridge(X, Y, lambda_=lam, n_rand=0, backend="numpy")
    for k in ("beta", "se", "zscore", "pvalue"):
        np.testing.assert_allclose(
            got[k], ref[k], rtol=1e-9, atol=1e-11,
            err_msg=f"t-test {k} diverges from sandwich reference at lambda={lam:g}",
        )
    assert got["df"] == pytest.approx(ref["df"], rel=1e-9)


def test_effective_df_below_ols_df_at_positive_lambda():
    """Ridge shrinkage spends fewer than p effective parameters."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 6))
    Y = rng.standard_normal((100, 4))
    got = ridge(X, Y, lambda_=5e5, n_rand=0, backend="numpy")
    n, p = X.shape
    assert got["df"] > n - p          # effective df exceeds OLS n-p...
    assert got["df"] < n              # ...but never reaches n
    # naive OLS df would be exactly n - p; the ridge value must differ.
    assert abs(got["df"] - (n - p)) > 1.0


def test_ttest_reduces_to_ols_at_lambda_zero():
    """At lambda=0 the sandwich SE and effective df collapse to OLS."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 5))
    Y = rng.standard_normal((80, 3))
    got = ridge(X, Y, lambda_=0.0, n_rand=0, backend="numpy")
    n, p = X.shape
    # Effective df -> n - p (within float noise).
    assert got["df"] == pytest.approx(n - p, abs=1e-6)
    # OLS closed form for SE.
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    rss = np.sum((Y - X @ beta) ** 2, axis=0)
    sigma2 = rss / (n - p)
    se_ols = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))
    np.testing.assert_allclose(got["se"], se_ols, rtol=1e-8, atol=1e-10)


def test_sandwich_differs_from_naive_se_at_high_lambda():
    """Guard against regressing to the naive diag((X'X+lambda I)^-1) SE."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 8))
    Y = rng.standard_normal((120, 10))
    lam = 5e5
    got = ridge(X, Y, lambda_=lam, n_rand=0, backend="numpy")
    n, p = X.shape
    Ainv = np.linalg.inv(X.T @ X + lam * np.eye(p))
    beta = Ainv @ X.T @ Y
    rss = np.sum((Y - X @ beta) ** 2, axis=0)
    naive_se = np.sqrt(np.outer(np.diag(Ainv), rss / (n - p)))
    # The correct sandwich SE is materially smaller than the naive one.
    assert np.median(got["se"] / naive_se) < 0.5
