# Advanced API: Low-Level Functions

The high-level functions handle gene subsetting, scaling, centering, and
streaming output automatically. If you need more control — for example, to
pass a pre-processed sparse matrix directly — use the low-level functions:

| Function | Input | Best for |
|----------|-------|----------|
| `ridge()` | Dense or sparse Y (fits in memory) | Small-to-medium datasets, single-call |
| `ridge_batch()` | Dense or sparse Y (any size) | Large datasets, streaming output |

Both functions support the same `backend` options (`'numpy'`, `'cupy'`,
`'auto'`), `sparse_mode`, `col_center`, `col_scale`, and produce identical
results for the same inputs.

## Dense vs sparse inputs

How normalization is handled depends on the input format:

- **Dense (NumPy array):** You must z-score normalize Y yourself before
  calling, since neither `ridge()` nor `ridge_batch()` scans the full dense
  array upfront. The `col_center`/`col_scale` flags are ignored.
- **Sparse (`scipy.sparse` matrix):** Column statistics are computed
  efficiently from the sparse structure, then normalization is applied
  in-flight. Use `col_center` and `col_scale` to control which normalization
  steps are applied (both default to `True`).

## Column normalization flags for sparse Y

When Y is sparse, the `col_center` and `col_scale` parameters control
in-flight normalization. For each output element (i, j), where μⱼ and σⱼ
are the mean and standard deviation of column j of Y:

| `col_center` | `col_scale` | Element (i, j) formula | Description |
|---|---|---|---|
| `True` | `True` | `Σₖ Tᵢₖ(Yₖⱼ − μⱼ) / σⱼ` | Full z-scoring (default) |
| `True` | `False` | `Σₖ Tᵢₖ(Yₖⱼ − μⱼ)` | Mean-center only |
| `False` | `True` | `Σₖ Tᵢₖ Yₖⱼ / σⱼ` | Scale only |
| `False` | `False` | `Σₖ Tᵢₖ Yₖⱼ` | Raw projection |

These flags work with both `ridge()` and `ridge_batch()`, and with both
`sparse_mode=True` and `sparse_mode=False`.

## Examples

```python
from secactpy import ridge, ridge_batch

# --- ridge() with sparse Y ---
# Full in-flight z-scoring (matches pre-scaled dense input)
result = ridge(X, Y_sparse, sparse_mode=True)

# Raw projection without normalization
result = ridge(X, Y_sparse, sparse_mode=True,
               col_center=False, col_scale=False)

# --- ridge_batch() with sparse Y ---
# Sparse end-to-end with in-flight normalization and row centering
result = ridge_batch(
    X, Y_sparse,
    batch_size=5000,
    sparse_mode=True,    # keep Y sparse during T @ Y
    row_center=True,     # apply row-mean centering in-flight
    col_center=True,     # subtract column means (default)
    col_scale=True       # divide by column stds (default)
)

# Raw sparse projection (no column normalization)
result = ridge_batch(
    X, Y_sparse,
    batch_size=5000,
    col_center=False, col_scale=False
)
```

## Disabling automatic sparse scaling

If you do not want automatic sparse scaling, you can either set
`col_center=False, col_scale=False`, or convert to dense and normalize
however you like:

```python
from secactpy import ridge_batch

# Option 1: Disable in-flight normalization
result = ridge_batch(X, Y_sparse, batch_size=5000,
                     col_center=False, col_scale=False)

# Option 2: Convert to dense, apply your own processing
Y_dense = Y_sparse.toarray().astype(np.float64)
# ... apply your own normalization (or skip it) ...
result = ridge_batch(X, Y_dense, batch_size=5000)
```
