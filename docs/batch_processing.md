# Batch Processing

## What is batch processing?

By default, SecActPy loads the entire expression matrix into memory and runs
ridge regression on all samples at once. This works well for most datasets, but
for large-scale analyses (e.g., 50,000+ single cells or spatial spots) the
memory required for permutation testing can exceed available RAM or GPU memory.

Batch processing splits the work into smaller pieces. The expensive projection
matrix `T = (X'X + λI)^{-1} X'` is computed **once** from the signature, then
samples are processed in chunks of `batch_size` at a time. Each chunk goes
through the full permutation-testing pipeline independently, and partial results
are concatenated at the end. **The final output is mathematically identical** to
processing all samples at once — only peak memory usage is reduced.

All three high-level functions support `batch_size` and `output_path`:
- `secact_activity_inference()` — bulk RNA-seq
- `secact_activity_inference_scrnaseq()` — scRNA-seq
- `secact_activity_inference_st()` — spatial transcriptomics

Set `batch_size` to enable it:

```python
# Without batch processing: all samples at once (default)
result = secact_activity_inference(expr_df, ...)

# With batch processing: 5000 samples per chunk
result = secact_activity_inference(expr_df, ..., batch_size=5000)

# Works the same way for scRNA-seq and ST:
result = secact_activity_inference_scrnaseq(adata, ..., batch_size=5000)
result = secact_activity_inference_st(adata, ..., batch_size=5000)
```

## In-memory vs streaming output

By default, batch results are accumulated in memory and returned as a dictionary
of DataFrames — this is the **in-memory** mode. You get back a `dict` with
`result['zscore']`, `result['pvalue']`, etc., just like the non-batched case.

For very large datasets, even the **output** matrices (beta, zscore, pvalue,
se — each of shape n_proteins × n_samples) may not fit in memory. **Streaming
output** solves this: set `output_path` to write each batch's results directly
to an HDF5 file on disk as it completes. The function returns `None` in this
mode — no results are held in memory. You load them back from the file when
needed. All three high-level functions support this.

| Mode | Parameter | Return value | Memory for output |
|------|-----------|--------------|-------------------|
| In-memory (default) | `output_path=None` | `dict` of DataFrames | All results in RAM |
| Streaming | `output_path="results.h5ad"` | `None` | Only one batch at a time |

```python
# Streaming works with any high-level function:
secact_activity_inference(..., batch_size=5000, output_path="bulk_results.h5ad")
secact_activity_inference_scrnaseq(..., batch_size=5000, output_path="sc_results.h5ad")
secact_activity_inference_st(..., batch_size=5000, output_path="st_results.h5ad")
```

## Example: batch processing with `secact_activity_inference`

`secact_activity_inference` handles gene subsetting, z-score normalization,
signature grouping, and row expansion automatically — you just pass your
expression data and set `batch_size`.

```bash
# Download example data (788 OV CD4 T cells, 34 MB)
wget https://zenodo.org/records/18520356/files/OV_scRNAseq_CD4.h5ad
```

```python
from secactpy import secact_activity_inference
import anndata as ad

# Load multi-sample expression data
adata = ad.read_h5ad("OV_scRNAseq_CD4.h5ad")

# --- In-memory mode (default) ---
# Results are returned as a dict of DataFrames
result = secact_activity_inference(
    adata.to_df().T,         # genes × cells DataFrame
    is_differential=False,   # center by row means across samples
    batch_size=200,          # process 200 cells per batch
    verbose=True
)
print(result['zscore'].head())  # (proteins × cells) DataFrame

# --- Streaming mode ---
# Results are written to disk; function returns None
secact_activity_inference(
    adata.to_df().T,
    is_differential=False,
    batch_size=200,
    output_path="results.h5ad",       # write here instead of returning
    output_compression="gzip",        # compress on disk (default)
    verbose=True
)
# Load results back when needed:
import h5py
with h5py.File("results.h5ad", "r") as f:
    zscore = f['zscore'][:]           # NumPy array (proteins × cells)
```

## Sparse mode for memory-efficient processing

By default, sparse Y matrices are converted to dense before matrix
multiplication. For very large, highly sparse datasets (e.g., scRNA-seq with
100k+ cells at <5% density), this can require hundreds of GB of RAM.

Setting `sparse_mode=True` keeps Y sparse throughout the entire pipeline.
Instead of densifying Y and computing `T @ Y_dense`, it uses the algebraic
identity `(Y.T @ T.T).T` to perform the multiplication with Y in sparse format.
Column z-scoring and row-mean centering are applied as lightweight corrections
on the small output matrix, never on Y itself.

All three high-level functions support `sparse_mode`:

```python
# scRNA-seq: sparse CPM → log2 → ridge, Y never densified
result = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="cell_type",
    is_single_cell_level=True,
    batch_size=5000,
    sparse_mode=True,   # keep Y sparse end-to-end
    verbose=True
)

# Spatial transcriptomics: same sparse pipeline
result = secact_activity_inference_st(
    adata,
    batch_size=5000,
    sparse_mode=True,
    verbose=True
)
```

When `sparse_mode=True`, the scrnaseq and ST functions bypass the standard
dense normalization pipeline. CPM normalization and log2 transform are applied
directly on the sparse matrix (both are zero-preserving), and row-mean
centering + column z-scoring are handled in-flight by `ridge_batch`.

| | Default (`sparse_mode=False`) | `sparse_mode=True` |
|---|---|---|
| Memory | Allocates full dense Y | Y stays sparse |
| Speed (<5% density) | Baseline | ~1.8x faster |
| Speed (5-10% density) | Baseline | ~25% slower |
| Results | Identical | Identical |
