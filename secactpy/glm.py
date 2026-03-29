"""
Generalized linear models for SecActPy.

Pure Python/NumPy implementation of logistic regression with optional
Firth bias correction and GPU acceleration via CuPy.

Replaces the C extension in data_significance. The algorithm follows
the same Newton-Raphson procedure as glm.c, using Cholesky decomposition
of the information matrix X'WX.

Reference: Firth (1993) Bias Reduction of Maximum Likelihood Estimates.
           Biometrika 80(1):27-38.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

__all__ = ["logistic_regression", "logit"]

EPS = 1e-10

# Check CuPy availability (same pattern as ridge.py)
try:
    import cupy as cp
    import cupy.linalg as cp_linalg
    # Verify GPU is actually usable (not just importable)
    cp.cuda.runtime.getDevice()
    CUPY_AVAILABLE = True
except (ImportError, Exception):
    cp = None
    cp_linalg = None
    CUPY_AVAILABLE = False


def logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_delta: float = 5.0,
    max_iter: int = 200,
    firth: bool = False,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Logistic regression with optional Firth bias correction.

    Newton-Raphson with Cholesky-decomposed information matrix,
    matching data_significance.logit() behavior.

    Parameters
    ----------
    X : array (n_samples, n_features)
        Design matrix. Should include intercept column if desired.
    y : array (n_samples,)
        Binary outcome (0 or 1).
    tol : float
        Convergence tolerance on gradient norm.
    max_delta : float
        Maximum relative step size per iteration.
    max_iter : int
        Maximum Newton-Raphson iterations.
    firth : bool
        Apply Firth bias correction (penalized likelihood).
    backend : {"auto", "numpy", "cupy"}
        Computation backend. "auto" uses GPU if available.
    verbose : bool
        Print convergence info.

    Returns
    -------
    dict with keys:
        beta : array (n_features,)    -- coefficients
        stderr : array (n_features,)  -- standard errors
        z : array (n_features,)       -- Wald z-statistics
        pvalue : array (n_features,)  -- two-sided p-values
        converged : bool
        n_iter : int
        method : str                  -- "numpy" or "cupy"
    """
    # Resolve backend
    if backend == "auto":
        use_gpu = CUPY_AVAILABLE
    elif backend == "cupy":
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy backend requested but not available")
        use_gpu = True
    else:
        use_gpu = False

    if use_gpu:
        return _logistic_regression_gpu(
            X, y, tol=tol, max_delta=max_delta, max_iter=max_iter,
            firth=firth, verbose=verbose,
        )
    return _logistic_regression_cpu(
        X, y, tol=tol, max_delta=max_delta, max_iter=max_iter,
        firth=firth, verbose=verbose,
    )


def _logistic_regression_cpu(
    X, y, *, tol, max_delta, max_iter, firth, verbose,
) -> dict[str, np.ndarray]:
    """CPU (NumPy/SciPy) implementation."""
    from scipy import linalg

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape

    if n != len(y):
        raise ValueError(f"X has {n} rows but y has {len(y)} elements")

    beta = np.zeros(p)
    converged = False
    n_iter = max_iter

    for it in range(max_iter):
        eta = X @ beta
        P = _sigmoid_np(eta)
        W = np.maximum(P * (1 - P), EPS)
        error = y - P

        sqrt_W = np.sqrt(W)
        wX = X * sqrt_W[:, None]
        I = wX.T @ wX

        try:
            L = linalg.cholesky(I, lower=True)
        except linalg.LinAlgError:
            if verbose:
                print("Cholesky failed on information matrix")
            return _fail_result(p, it, "numpy")

        I_inv = linalg.cho_solve((L, True), np.eye(p))

        if firth:
            H_t = wX @ I_inv
            H_diag = np.sum(H_t * wX, axis=1)
            error = error - H_diag * (P - 0.5)

        U = X.T @ error

        if np.linalg.norm(U) < tol:
            converged = True
            n_iter = it
            break

        delta = linalg.cho_solve((L, True), U)
        _apply_step_control(delta, beta, max_delta)
        beta = beta + delta

    return _compute_stats(beta, I_inv, converged, n_iter, verbose, max_iter, U, "numpy")


def _logistic_regression_gpu(
    X, y, *, tol, max_delta, max_iter, firth, verbose,
) -> dict[str, np.ndarray]:
    """GPU (CuPy) implementation — same algorithm, all ops on device."""
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64).ravel()
    n, p = X_np.shape

    if n != len(y_np):
        raise ValueError(f"X has {n} rows but y has {len(y_np)} elements")

    if verbose:
        print(f"  GPU logistic regression: {n} samples, {p} features")

    # Transfer to GPU
    X_g = cp.asarray(X_np)
    y_g = cp.asarray(y_np)
    beta = cp.zeros(p, dtype=cp.float64)
    eye_p = cp.eye(p, dtype=cp.float64)

    converged = False
    n_iter = max_iter

    for it in range(max_iter):
        eta = X_g @ beta
        P = _sigmoid_cp(eta)
        W = cp.maximum(P * (1 - P), EPS)
        error = y_g - P

        sqrt_W = cp.sqrt(W)
        wX = X_g * sqrt_W[:, None]
        I = wX.T @ wX

        try:
            L = cp_linalg.cholesky(I)  # CuPy cholesky returns lower
        except cp_linalg.LinAlgError:
            if verbose:
                print("Cholesky failed on GPU")
            return _fail_result(p, it, "cupy")

        # Reuse Cholesky factor for solves
        I_inv = cp_linalg.solve(L @ L.T, eye_p)

        if firth:
            H_t = wX @ I_inv
            H_diag = cp.sum(H_t * wX, axis=1)
            error = error - H_diag * (P - 0.5)

        U = X_g.T @ error

        if float(cp.linalg.norm(U)) < tol:
            converged = True
            n_iter = it
            break

        delta = cp_linalg.solve(L @ L.T, U)

        # Step control on GPU
        norm_delta = float(cp.linalg.norm(delta))
        norm_beta = float(cp.linalg.norm(beta))
        step_ratio = (norm_delta + 1) / (norm_beta + 1)
        if step_ratio > max_delta:
            delta *= max_delta / step_ratio

        beta = beta + delta

    # Transfer results back to CPU
    beta_np = cp.asnumpy(beta)
    I_inv_np = cp.asnumpy(I_inv)
    U_np = cp.asnumpy(U) if not converged else None

    return _compute_stats(beta_np, I_inv_np, converged, n_iter, verbose, max_iter, U_np, "cupy")


# ── Shared helpers ────────────────────────────────────────────────────────


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (NumPy)."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def _sigmoid_cp(x):
    """Numerically stable sigmoid (CuPy)."""
    return cp.where(
        x >= 0,
        1 / (1 + cp.exp(-x)),
        cp.exp(x) / (1 + cp.exp(x)),
    )


def _apply_step_control(delta, beta, max_delta):
    """In-place step size limiting."""
    norm_delta = np.linalg.norm(delta)
    norm_beta = np.linalg.norm(beta)
    step_ratio = (norm_delta + 1) / (norm_beta + 1)
    if step_ratio > max_delta:
        delta *= max_delta / step_ratio


def _fail_result(p: int, n_iter: int, method: str) -> dict:
    return {
        "beta": np.zeros(p),
        "stderr": np.zeros(p),
        "z": np.zeros(p),
        "pvalue": np.ones(p),
        "converged": False,
        "n_iter": n_iter,
        "method": method,
    }


def _compute_stats(beta, I_inv, converged, n_iter, verbose, max_iter, U, method):
    """Compute z-scores and p-values from final estimates."""
    var_beta = np.diag(I_inv)
    stderr = np.sqrt(np.maximum(var_beta, 0))
    z = np.where(stderr > 0, beta / stderr, 0.0)
    pvalue = np.where(stderr > 0, 2 * (1 - stats.norm.cdf(np.abs(z))), 1.0)

    if verbose and not converged:
        norm = np.linalg.norm(U) if U is not None else float("nan")
        print(f"Did not converge after {max_iter} iterations, "
              f"gradient norm = {norm:.6f}")

    return {
        "beta": beta,
        "stderr": stderr,
        "z": z,
        "pvalue": pvalue,
        "converged": converged,
        "n_iter": n_iter,
        "method": method,
    }


# ── High-level API ────────────────────────────────────────────────────────


def logit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-5,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    firth: bool = True,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """High-level logistic regression matching CytoSig/CIDE calling convention.

    Parameters
    ----------
    X : array (n_samples, n_features)
        Design matrix.
    y : array (n_samples,)
        Binary outcome.
    alpha : float
        Not used (kept for API compatibility with data_significance).
    alternative : str
        P-value sidedness.
    firth : bool
        Apply Firth correction.
    backend : {"auto", "numpy", "cupy"}
        Computation backend.
    verbose : bool
        Print convergence info.

    Returns
    -------
    dict with beta, stderr, z, pvalue arrays.
    """
    result = logistic_regression(
        X, y, firth=firth, backend=backend, verbose=verbose,
    )

    if alternative == "less":
        result["pvalue"] = stats.norm.cdf(result["z"])
    elif alternative == "greater":
        result["pvalue"] = 1 - stats.norm.cdf(result["z"])

    return result
