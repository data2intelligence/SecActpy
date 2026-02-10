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
in-flight normalization.

**Notation:**

| Symbol | Shape | Definition |
|--------|-------|------------|
| T | (m, p) | Projection matrix: `(X'X + λI)⁻¹ X'`, where X is the signature matrix |
| Y | (p, n) | Expression matrix (genes × samples), sparse |
| Tᵢₖ | scalar | Element at row i, column k of T |
| Yₖⱼ | scalar | Element at row k, column j of Y |
| μⱼ | scalar | Mean of column j of Y: `μⱼ = (1/p) Σₖ Yₖⱼ` |
| σⱼ | scalar | Standard deviation of column j of Y |
| μ | (n,) | Vector of all column means |
| σ | (n,) | Vector of all column stds |
| Σₖ | — | Summation over k = 1, …, p (gene axis) |

**Formulas:**

| `col_center` | `col_scale` | Element (i, j) formula | Python (vectorized) | Description |
|---|---|---|---|---|
| `True` | `True` | `Σₖ Tᵢₖ(Yₖⱼ − μⱼ) / σⱼ` | `(T @ Y - T.sum(1)[:, None] * μ) / σ` | Full z-scoring (default) |
| `True` | `False` | `Σₖ Tᵢₖ(Yₖⱼ − μⱼ)` | `T @ Y - T.sum(1)[:, None] * μ` | Column mean-center only |
| `False` | `True` | `Σₖ Tᵢₖ Yₖⱼ / σⱼ` | `T @ Y / σ` | Column scale only |
| `False` | `False` | `Σₖ Tᵢₖ Yₖⱼ` | `T @ Y` | Raw projection |

**Broadcasting:** `μ` and `σ` are 1-D vectors of length n (one value per
column of Y). `T @ Y` produces an (m × n) matrix and the vector operations
broadcast across it:

- **`/ σ`** — `σ` has shape `(n,)`. NumPy broadcasts it as a row vector,
  dividing each column j of the matrix by `σⱼ`. This scales every element
  in column j by the same scalar.
- **`* μ`** — same broadcasting: `μ` has shape `(n,)` and multiplies each
  column j by `μⱼ`.
- **`T.sum(1)[:, None]`** — row sums of T, shape `(m, 1)`. The `[:, None]`
  reshapes it into a column vector so that `T.sum(1)[:, None] * μ` produces
  an (m × n) outer product, where element (i, j) equals `(Σₖ Tᵢₖ) · μⱼ`.
  This is the centering correction term subtracted from `T @ Y`.

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
