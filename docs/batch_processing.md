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

## Streaming H5AD for very large datasets (>5M cells)

Batch processing and sparse mode assume the full expression matrix fits in
memory. For datasets with >5M cells (e.g., 6.5M-cell atlases), even the sparse
matrix can exceed available RAM (60+ GB for the sparse data alone, 200+ GB peak
after transposition, CPM normalization, and log2 transform).

**Streaming H5AD** (`streaming=True`) solves this by reading the H5AD file in
chunks via h5py, never loading the full matrix. It uses a two-pass algorithm:

1. **Pass 1 (Statistics)**: For each chunk of cells, read CSR rows from H5AD,
   transpose, CPM-normalize, log2-transform, subset to signature genes, and
   accumulate row sums, column sums, and column sum-of-squares. The chunk data
   is then discarded.

2. **Between passes**: Compute row means, the projection matrix T, and the
   permutation table (one-time cost, same as standard batch mode).

3. **Pass 2 (Inference)**: Re-read each chunk, apply the same normalization,
   compute per-chunk cross terms (`Y_chunk.T @ row_means`), derive per-cell
   sigma, and feed sub-batches to the existing sparse batch inference functions.
   Results are written to disk via `StreamingResultWriter`.

### Usage

```python
from secactpy import secact_activity_inference_scrnaseq

# scRNA-seq: process 6.5M cells with ~3 GB peak memory
result = secact_activity_inference_scrnaseq(
    "cima_atlas.h5ad",               # must be a file path, not AnnData
    cell_type_col="cell_type",
    is_single_cell_level=True,       # required for streaming
    streaming=True,                  # enable two-pass chunk reading
    streaming_chunk_size=50_000,     # cells per chunk (default)
    output_path="cima_results.h5ad", # stream results to disk
    sparse_mode=True,                # keep chunks sparse (recommended)
    backend="auto",                  # GPU if available
    verbose=True,
)
# result is None when output_path is set; load with h5py or anndata
```

```python
from secactpy import secact_activity_inference_st

# Spatial transcriptomics: same interface
result = secact_activity_inference_st(
    "large_spatial.h5ad",
    streaming=True,
    streaming_chunk_size=50_000,
    output_path="spatial_results.h5ad",
    verbose=True,
)
```

### Requirements

- `adata` / `input_data` must be a **file path** (str or Path), not an in-memory AnnData object
- `is_single_cell_level=True` (scRNA-seq) or `is_spot_level=True` (ST)
- The H5AD file must store X in **sparse** (CSR or CSC) format
- h5py must be installed

### Memory comparison

| Component | Standard (5M cells) | Streaming (5M cells) |
|-----------|---------------------|----------------------|
| Full sparse X | ~60 GB | 0 |
| Transpose + normalize | ~120 GB peak | 0 |
| One chunk (50K cells) | — | ~1 GB |
| Stats accumulators | — | ~80 MB |
| T matrix + permutations | ~25 MB | ~25 MB |
| **Peak** | **~200 GB** | **~3 GB** |

### Low-level API

For advanced users, the streaming components are also available directly:

```python
from secactpy import H5ADChunkReader, ridge_batch_streaming

# Read H5AD in chunks
with H5ADChunkReader("data.h5ad", chunk_size=50_000) as reader:
    print(f"Shape: {reader.n_obs} cells × {reader.n_vars} genes")
    var_names = reader.read_var_names()
    obs_names = reader.read_obs_names()
    for start, end, chunk_csr in reader.iter_chunks():
        print(f"  Chunk [{start}:{end}]: {chunk_csr.shape}")
```
