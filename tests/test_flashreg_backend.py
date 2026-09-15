"""secactpy's optional flashreg accelerator.

SecAct R dispatches `backend="auto"` to FlashReg::ridge when that package is installed
and falls back to its own pure-R loop otherwise. secactpy mirrors it: flashregpy's
C+OpenMP kernel when importable, its own NumPy path when not. `pip install secactpy[fast]`

The agreement target is ~1e-16 absolute on beta, NOT bit-identity -- a threaded reduction
sums in a different order, and double addition is not associative. See
flashreg/tests/test_ridge_parity.py for the measurement that establishes this.
"""
from __future__ import annotations
import numpy as np
import pytest

from secactpy.ridge import FLASHREG_AVAILABLE, resolve_backend, ridge

ATOL_BETA = 1e-12
ATOL_Z = 1e-10


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(1423)
    return rng.normal(size=(600, 40)), rng.normal(size=(600, 6))


def test_flashreg_listed_and_probed():
    from secactpy.ridge import _BACKENDS
    assert "flashreg" in _BACKENDS
    assert isinstance(FLASHREG_AVAILABLE, bool)


def test_missing_flashreg_raises_actionable_error():
    if FLASHREG_AVAILABLE:
        pytest.skip("flashregpy is installed here")
    with pytest.raises(ImportError, match=r"secactpy\[fast\]"):
        resolve_backend("flashreg")


@pytest.mark.skipif(not FLASHREG_AVAILABLE, reason="flashregpy not installed")
def test_auto_prefers_flashreg_over_numpy_on_cpu():
    """Only when no GPU path is available -- GPU still outranks it."""
    from secactpy.ridge import CUDA_NATIVE_AVAILABLE, CUPY_AVAILABLE
    got = resolve_backend("auto")
    if CUDA_NATIVE_AVAILABLE or CUPY_AVAILABLE:
        assert got in ("cuda_native", "cupy")
    else:
        assert got == "flashreg", "auto should prefer the accelerator over numpy"


@pytest.mark.skipif(not FLASHREG_AVAILABLE, reason="flashregpy not installed")
@pytest.mark.parametrize("field,atol", [("beta", ATOL_BETA), ("zscore", ATOL_Z)])
def test_flashreg_matches_numpy(xy, field, atol):
    X, Y = xy
    kw = dict(lambda_=5e5, n_rand=100, seed=0, rng_method="mt19937")
    a = ridge(X, Y, backend="numpy", **kw)[field]
    b = ridge(X, Y, backend="flashreg", **kw)[field]
    d = float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    print(f"\n  {field}: numpy vs flashreg max |diff| = {d:.3e} (gate {atol:g})")
    assert d <= atol, f"{field} max |diff| {d:.3e} exceeds {atol:g}"


@pytest.mark.skipif(not FLASHREG_AVAILABLE, reason="flashregpy not installed")
def test_flashreg_returns_the_expected_contract(xy):
    X, Y = xy
    r = ridge(X, Y, lambda_=5e5, n_rand=50, seed=0, backend="flashreg")
    for k in ("beta", "se", "zscore", "pvalue"):
        assert k in r, f"missing {k}"
        assert np.shape(r[k]) == (X.shape[1], Y.shape[1])
    assert r["method"] == "flashreg"
