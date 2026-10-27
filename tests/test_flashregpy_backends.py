"""Backends: SecActpy's own numpy/cupy, and omp/cuda_native through optional flashregpy.

Without flashregpy, "omp" and "cuda_native" must raise ImportError naming the install
and "auto" must never pick them. With it, both must reproduce the numpy backend: same
permutation table from (rng_method, seed), so beta/se/zscore agree to rounding and the
permutation p-values are identical.
"""
import importlib

import numpy as np
import pytest
import scipy.sparse as sps


# `secactpy.ridge` the attribute is the re-exported function; load the modules by name.
R = importlib.import_module("secactpy.ridge")
B = importlib.import_module("secactpy.batch")
ridge, resolve_backend = R.ridge, R.resolve_backend


def _usable(backend):
    return R._flashregpy_backend_error(backend) is None


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 6))
    Y = rng.standard_normal((300, 25))
    Ys = sps.random(300, 40, density=0.1, format="csc", random_state=3,
                    data_rvs=lambda k: rng.poisson(3.0, k) + 1.0)
    return X, Y, Ys


def test_flashreg_is_not_a_backend_name():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_backend("flashreg")


@pytest.mark.parametrize("backend", ["omp", "cuda_native"])
def test_missing_flashregpy_names_the_install(monkeypatch, backend):
    monkeypatch.setattr(R, "FLASHREGPY_AVAILABLE", False)
    monkeypatch.setattr(R, "_flashregpy", None)
    with pytest.raises(ImportError, match=r"secactpy\[fast\]"):
        resolve_backend(backend)


def test_auto_without_flashregpy_is_cupy_or_numpy(monkeypatch):
    monkeypatch.setattr(R, "FLASHREGPY_AVAILABLE", False)
    monkeypatch.setattr(R, "_flashregpy", None)
    assert resolve_backend("auto") == ("cupy" if R.CUPY_AVAILABLE else "numpy")


def test_auto_order_prefers_flashregpy_gpu_then_cupy_then_omp(monkeypatch):
    monkeypatch.setattr(R, "_flashregpy_auto_choice", lambda: "cuda_native")
    assert resolve_backend("auto") == "cuda_native"
    monkeypatch.setattr(R, "_flashregpy_auto_choice", lambda: "omp")
    monkeypatch.setattr(R, "CUPY_AVAILABLE", True)
    assert resolve_backend("auto") == "cupy"
    monkeypatch.setattr(R, "CUPY_AVAILABLE", False)
    assert resolve_backend("auto") == "omp"
    monkeypatch.setattr(R, "_flashregpy_auto_choice", lambda: None)
    assert resolve_backend("auto") == "numpy"


def test_dispatch_reaches_flashregpy_with_secactpy_arguments(monkeypatch, data):
    """The call flashregpy receives, checked without flashregpy installed."""
    X, Y, Ys = data
    seen = []

    class Fake:
        @staticmethod
        def ridge(X, Y, **kw):
            seen.append((sps.issparse(Y), kw))
            z = np.zeros((X.shape[1], Y.shape[1]))
            return {"beta": z, "se": z, "zscore": z, "pvalue": z}

    monkeypatch.setattr(R, "_flashregpy", Fake)
    monkeypatch.setattr(R, "resolve_backend", lambda b: b)
    ridge(X, Y, lambda_=5.0, n_rand=20, seed=3, backend="omp", rng_method="gsl")
    ridge(X, Ys, lambda_=5.0, n_rand=20, seed=3, backend="cuda_native",
          sparse_mode=True, col_scale=False)
    (sp0, kw0), (sp1, kw1) = seen
    assert not sp0 and kw0["backend"] == "omp" and kw0["n_perm"] == 20
    assert kw0["seed"] == 3 and kw0["rng_method"] == "gsl" and kw0["precision"] == "fp64"
    assert sp1 and kw1["backend"] == "cuda_native" and kw1["sparse_mode"] is True
    assert kw1["col_center"] is True and kw1["col_scale"] is False


def test_ttest_stays_on_numpy_for_flashregpy_backends(monkeypatch, data):
    X, Y, _ = data
    monkeypatch.setattr(R, "resolve_backend", lambda b: b)
    monkeypatch.setattr(R, "_flashregpy", None)   # must not be touched
    out = ridge(X, Y, lambda_=5.0, n_rand=0, backend="omp")
    ref = ridge(X, Y, lambda_=5.0, n_rand=0, backend="numpy")
    assert out["method"] == "omp_ttest"
    for k in ("beta", "se", "zscore", "pvalue"):
        np.testing.assert_array_equal(out[k], ref[k])


@pytest.mark.parametrize("backend", ["omp", "cuda_native"])
@pytest.mark.parametrize("rng_method", ["srand", "gsl"])
@pytest.mark.parametrize("sparse", [False, True])
def test_flashregpy_backend_matches_numpy(data, backend, rng_method, sparse):
    if not _usable(backend):
        pytest.skip(R._flashregpy_backend_error(backend))
    X, Y, Ys = data
    Yin = Ys if sparse else Y
    kw = dict(lambda_=5.0, n_rand=60, seed=3, rng_method=rng_method, sparse_mode=sparse)
    ref = ridge(X, Yin, backend="numpy", **kw)
    out = ridge(X, Yin, backend=backend, **kw)
    assert out["method"] == backend
    for k, tol in (("beta", 1e-10), ("se", 1e-10), ("zscore", 1e-8)):
        np.testing.assert_allclose(out[k], ref[k], rtol=0, atol=tol, err_msg=k)
    np.testing.assert_array_equal(out["pvalue"], ref["pvalue"])


def test_batch_auto_never_picks_flashregpy(monkeypatch):
    monkeypatch.setattr(R, "_flashregpy_auto_choice", lambda: "omp")
    assert B._resolve_batch_backend("auto") == ("cupy" if B.CUPY_AVAILABLE else "numpy")


def test_batch_explicit_flashregpy_backend_warns_and_says_what_runs(monkeypatch):
    monkeypatch.setattr(B, "resolve_backend", lambda b: b)
    monkeypatch.setattr(B, "CUPY_AVAILABLE", False)
    with pytest.warns(UserWarning, match="runs on 'numpy'"):
        assert B._resolve_batch_backend("omp") == "numpy"
    with pytest.warns(UserWarning, match="runs on 'numpy'"):
        assert B._resolve_batch_backend("cuda_native") == "numpy"
