"""The cached permutation table must come from the generator that was asked for.

Before the fix, get_cached_inverse_perm_table always built its table with the C-stdlib
generator and keyed the file on (n, n_perm, seed) only, so ridge(..., rng_method="mt19937",
use_cache=True) silently ran on the srand table and returned different z-scores from the
uncached mt19937 call.
"""
import numpy as np
import pytest

from secactpy.rng import (CStdlibRNG, GSLRNG, get_cached_inverse_perm_table,
                          get_cached_perm_table)
from secactpy.ridge import ridge


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SECACTPY_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("rng_method, cls", [("srand", CStdlibRNG), ("gsl", GSLRNG),
                                             ("mt19937", GSLRNG)])
def test_cached_table_is_the_requested_generator(cache_dir, rng_method, cls):
    n, n_perm = 57, 20
    want_inv = cls(0).inverse_permutation_table(n, n_perm)
    want = cls(0).permutation_table(n, n_perm)
    for _ in range(2):                                   # generated, then loaded from disk
        got_inv = get_cached_inverse_perm_table(n, n_perm, 0, verbose=False, rng_method=rng_method)
        got = get_cached_perm_table(n, n_perm, 0, verbose=False, rng_method=rng_method)
        np.testing.assert_array_equal(got_inv, want_inv)
        np.testing.assert_array_equal(got, want)


def test_generators_do_not_share_cache_files(cache_dir):
    n, n_perm = 57, 20
    a = get_cached_inverse_perm_table(n, n_perm, 0, verbose=False, rng_method="srand")
    b = get_cached_inverse_perm_table(n, n_perm, 0, verbose=False, rng_method="mt19937")
    assert not np.array_equal(a, b)
    names = sorted(p.name for p in cache_dir.iterdir())
    assert names == [f"inv_perm_n{n}_nperm{n_perm}_seed0.npy",
                     f"inv_perm_n{n}_nperm{n_perm}_seed0_mt19937.npy"]
    get_cached_inverse_perm_table(n, n_perm, 0, verbose=False, rng_method="gsl")
    assert len(list(cache_dir.iterdir())) == 2           # gsl and mt19937: one generator


def test_unknown_generator_is_refused(cache_dir):
    with pytest.raises(ValueError):
        get_cached_inverse_perm_table(10, 5, 0, verbose=False, rng_method="numpy")


@pytest.mark.parametrize("rng_method", ["srand", "mt19937"])
def test_ridge_cached_equals_uncached(cache_dir, rng_method):
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(80, 4)), rng.normal(size=(80, 3))
    kw = dict(lambda_=1e3, n_rand=50, seed=0, backend="numpy", rng_method=rng_method)
    z0 = ridge(X, Y, use_cache=False, **kw)["zscore"]
    z1 = ridge(X, Y, use_cache=True, **kw)["zscore"]
    np.testing.assert_array_equal(z0, z1)
