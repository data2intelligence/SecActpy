# Reproducibility

SecActPy supports three RNG backends via the `rng_method` parameter, each
offering different trade-offs between R compatibility and performance:

| `rng_method` | Description | Use case |
|---|---|---|
| `'srand'` | C stdlib `srand()`/`rand()` via ctypes | Match R SecAct/RidgeR results **on the same platform** |
| `'gsl'` | Mersenne Twister (GSL-compatible) | **Cross-platform** reproducibility within SecActPy |
| `'numpy'` | Native NumPy RNG (~70x faster) | Fast analysis when reproducibility with R is not needed |

## Matching R SecAct/RidgeR output

To reproduce R SecAct/RidgeR results on the same machine, use `rng_method='srand'`.
This uses the C standard library's `rand()` function, which matches R's internal
RNG on the same platform. Note that C `rand()` implementations differ across
operating systems, so results are platform-dependent.

```python
result = secact_activity_inference(
    expression,
    is_differential=True,
    sig_matrix="secact",
    lambda_=5e5,
    n_rand=1000,
    seed=0,
    rng_method="srand",  # Match R SecAct on same platform
)
```

## Cross-platform reproducibility (default)

The default RNG (`rng_method='gsl'`) uses a portable Mersenne Twister
implementation that produces identical results across all platforms (Linux,
macOS, Windows). This does **not** match R output, but guarantees consistent
SecActPy results everywhere.

```python
result = secact_activity_inference(
    expression,
    rng_method="gsl",  # Cross-platform reproducible (default)
)
```

## Fastest analysis

For maximum throughput when reproducibility is not required:

```python
result = secact_activity_inference(
    expression,
    rng_method="numpy",  # ~70x faster permutation generation
)
```
