# SecActPy

**Secreted Protein Activity Inference using Ridge Regression**

[![PyPI version](https://badge.fury.io/py/secactpy.svg)](https://pypi.org/project/secactpy/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/data2intelligence/SecActPy/actions/workflows/tests.yml/badge.svg)](https://github.com/data2intelligence/SecActPy/actions/workflows/tests.yml)
[![Docker](https://img.shields.io/docker/pulls/psychemistz/secactpy)](https://hub.docker.com/r/psychemistz/secactpy)

Python implementation of [SecAct](https://github.com/data2intelligence/SecAct) for inferring secreted protein activities from gene expression data.

**Key Features:**
- 🎯 **SecAct Compatible**: Produces identical results to the R SecAct/RidgeR package
- 🚀 **GPU Acceleration**: Optional CuPy backend for large-scale analysis
- 📊 **Million-Sample Scale**: Batch processing with streaming output for massive datasets
- 🔬 **Built-in Signatures**: Includes SecAct and CytoSig signature matrices
- 🧬 **Multi-Platform Support**: Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics (Visium, CosMx)
- 💾 **Smart Caching**: Optional permutation table caching for faster repeated analyses
- 🧮 **Sparse-Aware**: Automatic memory-efficient processing for sparse single-cell data

## Installation

> **Recommended**: Create a virtual environment before installing to avoid dependency conflicts with other packages.
>
> ```bash
> python -m venv secactpy-env
> source secactpy-env/bin/activate   # Linux/macOS
> # secactpy-env\Scripts\activate    # Windows
> ```

### From PyPI (Recommended)

```bash
# CPU Only
pip install secactpy

# With GPU Support (CUDA 11.x)
pip install "secactpy[gpu]"

# With GPU Support (CUDA 12.x)
pip install secactpy
pip install cupy-cuda12x
```

### From GitHub

```bash
# CPU Only
pip install git+https://github.com/data2intelligence/SecActpy.git

# With GPU Support (CUDA 11.x)
pip install "secactpy[gpu] @ git+https://github.com/data2intelligence/SecActpy.git"

# With GPU Support (CUDA 12.x)
pip install git+https://github.com/data2intelligence/SecActpy.git
pip install cupy-cuda12x
```

> **Important (CUDA 12.x users)**: Do **not** use the `[gpu]` extra on CUDA 12.x systems — it installs `cupy-cuda11x`, which conflicts with `cupy-cuda12x`. If you already installed with `[gpu]`, remove the conflicting package first:
> ```bash
> pip uninstall cupy-cuda11x
> pip install cupy-cuda12x
> ```

### Development Installation

```bash
git clone https://github.com/data2intelligence/SecActpy.git
cd SecActpy
pip install -e ".[dev]"
```

## Quick Start

### Example Data

Example datasets for all Quick Start tutorials are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18520356.svg)](https://doi.org/10.5281/zenodo.18520356)

| Example | Input File | Output File | Size |
|---------|-----------|-------------|------|
| Bulk RNA-seq | `Ly86-Fc_vs_Vehicle_logFC.txt` | `Ly86-Fc_vs_Vehicle_logFC_output.h5ad` | 0.5 MB |
| scRNA-seq (OV CD4 T cells) | `OV_scRNAseq_CD4.h5ad` | `OV_scRNAseq_ct_CD4_output.h5ad`, `OV_scRNAseq_sc_CD4_output.h5ad` | 34 MB |
| Visium ST (HCC) | `Visium_HCC_data.h5ad` | `Visium_HCC_output.h5ad` | 255 MB |
| CosMx (LIHC) | `LIHC_CosMx_data.h5ad` | `LIHC_CosMx_output.h5ad` | 3.0 GB |

Download all example files:

```bash
# Download individual files from Zenodo
wget https://zenodo.org/records/18520356/files/Ly86-Fc_vs_Vehicle_logFC.txt
wget https://zenodo.org/records/18520356/files/OV_scRNAseq_CD4.h5ad
wget https://zenodo.org/records/18520356/files/Visium_HCC_data.h5ad
wget https://zenodo.org/records/18520356/files/LIHC_CosMx_data.h5ad
```

### Example 1: Bulk RNA-seq

```python
import pandas as pd
from secactpy import secact_activity_inference

# Load differential expression data (genes × samples)
# Download: https://zenodo.org/records/18520356/files/Ly86-Fc_vs_Vehicle_logFC.txt
diff_expr = pd.read_csv("Ly86-Fc_vs_Vehicle_logFC.txt", sep=r"\s+", index_col=0)

# Run inference
result = secact_activity_inference(
    diff_expr,
    is_differential=True,
    sig_matrix="secact",  # or "cytosig"
    verbose=True
)

# Access results
activity = result['zscore']    # Activity z-scores
pvalues = result['pvalue']     # P-values
coefficients = result['beta']  # Regression coefficients
```

> **Note:** Set `is_differential=True` when the input is already log fold-change data.
> For single-column input with no control, row-mean centering is automatically skipped
> (it would produce all zeros).

### Example 2: scRNA-seq Analysis

```python
import anndata as ad
from secactpy import secact_activity_inference_scrnaseq

# Load scRNA-seq data (788 OV CD4 T cells, 3 subtypes)
# Download: https://zenodo.org/records/18520356/files/OV_scRNAseq_CD4.h5ad
adata = ad.read_h5ad("OV_scRNAseq_CD4.h5ad")

# Pseudo-bulk by cell type
result = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="Annotation",
    is_single_cell_level=False,
    verbose=True
)

# Single-cell level
result_sc = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="Annotation",
    is_single_cell_level=True,
    verbose=True
)
```

### Example 3: Spatial Transcriptomics

#### Visium (spot-level)

```python
from secactpy import secact_activity_inference_st

# Load Visium HCC data (3,415 spots)
# Download: https://zenodo.org/records/18520356/files/Visium_HCC_data.h5ad
result = secact_activity_inference_st(
    "Visium_HCC_data.h5ad",
    min_genes=1000,
    verbose=True
)

activity = result['zscore']  # (proteins × spots)
```

#### CosMx (single-cell spatial)

```python
import anndata as ad
from secactpy import secact_activity_inference_st

# Load CosMx LIHC data (443,515 cells, 1,000 genes, 12 cell types)
# Download: https://zenodo.org/records/18520356/files/LIHC_CosMx_data.h5ad
adata = ad.read_h5ad("LIHC_CosMx_data.h5ad")

# Cell-type resolution (pseudo-bulk by cell type)
result = secact_activity_inference_st(
    adata,
    cell_type_col="cellType",  # Column in adata.obs
    is_spot_level=False,        # Aggregate by cell type
    verbose=True
)

activity = result['zscore']  # (proteins × cell_types)
```

### Large-Scale Batch Processing

**What is batch processing?** By default, `secact_activity_inference` loads the
entire expression matrix into memory and runs ridge regression on all samples at
once. For large datasets (e.g., 50,000+ single cells), this can exceed available
RAM or GPU memory. Batch processing solves this: the expensive projection matrix
`T = (X'X + λI)^{-1} X'` is computed once, then samples are processed in
chunks of `batch_size` at a time. Each chunk produces partial results that are
concatenated at the end. The final output is identical — only peak memory usage
is reduced.

**How `secact_activity_inference` works:**

- **Input format:** Accepts a dense pandas DataFrame (or a file path). Sparse
  matrices are not supported — convert to a DataFrame first. The function
  handles all gene subsetting and z-score normalization internally, so you do
  not need to pre-process the data.
- **Batch size:** `batch_size=None` by default, meaning all samples are processed
  at once. Set `batch_size=5000` (or similar) to process in memory-bounded
  chunks when working with large datasets.
- **Streaming output:** Set `output_path="results.h5ad"` to write results
  directly to disk as each batch completes, instead of accumulating them in
  memory. The function returns `None` in this mode. This is useful when even
  the output matrices are too large for memory. Requires `batch_size`.

Internally, the function:
1. Finds overlapping genes between expression and signature matrices
2. Subsets both matrices to common genes
3. Z-score normalizes each column (mean=0, std=1)
4. Runs ridge regression (all at once, or in batches if `batch_size` is set)

```python
from secactpy import secact_activity_inference

# Load multi-sample expression data
# Download: https://zenodo.org/records/18520356/files/OV_scRNAseq_CD4.h5ad
import anndata as ad
adata = ad.read_h5ad("OV_scRNAseq_CD4.h5ad")

# Process all cells with batch processing (results in memory)
result = secact_activity_inference(
    adata.to_df().T,         # genes × cells DataFrame
    is_differential=False,   # center by row means across samples
    batch_size=5000,         # process 5000 cells per batch
    backend='cupy',          # GPU acceleration (or 'numpy' for CPU)
    verbose=True
)

# result['zscore'] is (proteins × samples)
print(result['zscore'].head())

# Stream results to disk for very large datasets
secact_activity_inference(
    adata.to_df().T,
    is_differential=False,
    batch_size=5000,
    output_path="results.h5ad",       # write here instead of returning
    output_compression="gzip",        # compress on disk (default)
    backend='cupy',
    verbose=True
)
# Returns None — load results back when needed:
import h5py
with h5py.File("results.h5ad", "r") as f:
    zscore = f['zscore'][:]
```

#### Advanced: `ridge_batch` for full control

The high-level `secact_activity_inference` handles gene subsetting, scaling,
centering, and streaming output automatically. If you need more control — for
example, to pass a sparse matrix directly or skip centering — use the
lower-level `ridge_batch` function.

**Dense vs sparse input.** `ridge_batch` accepts two input formats:
- **Dense (NumPy array):** You must z-score normalize Y yourself before calling,
  because the function processes Y in chunks and cannot compute whole-column
  statistics internally.
- **Sparse (`scipy.sparse` matrix):** Pass raw counts directly. The function
  computes column means and standard deviations from the full sparse matrix
  up front, then applies z-score normalization on-the-fly within each batch
  (without converting the entire matrix to dense). This is necessary because
  sparse matrices cannot be z-scored in place without losing sparsity.

If you do not want the automatic sparse scaling, convert to dense first and
normalize however you like:

```python
# Opt out of auto-scaling: convert sparse to dense, apply your own processing
Y_dense = Y_sparse.toarray().astype(np.float64)
# ... apply your own normalization ...
result = ridge_batch(X, Y_dense, batch_size=5000)
```

## API Reference

### High-Level Functions

| Function | Description |
|----------|-------------|
| `secact_activity_inference()` | Bulk RNA-seq inference |
| `secact_activity_inference_st()` | Spatial transcriptomics inference |
| `secact_activity_inference_scrnaseq()` | scRNA-seq inference |
| `load_signature(name='secact')` | Load built-in signature matrix |

### Core Functions

| Function | Description |
|----------|-------------|
| `ridge()` | Single-call ridge regression with permutation testing |
| `ridge_batch()` | Batch processing for large datasets (dense or sparse) |
| `estimate_batch_size()` | Estimate optimal batch size for available memory |
| `estimate_memory()` | Estimate memory requirements |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_matrix` | `"secact"` | Signature: "secact", "cytosig", or DataFrame |
| `lambda_` | `5e5` | Ridge regularization parameter |
| `n_rand` | `1000` | Number of permutations |
| `seed` | `0` | Random seed for reproducibility |
| `backend` | `'auto'` | 'auto', 'numpy', or 'cupy' |
| `use_cache` | `False` | Cache permutation tables to disk |

### ST-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_type_col` | `None` | Column in AnnData.obs for cell type |
| `is_spot_level` | `True` | If False, aggregate by cell type |
| `scale_factor` | `1e5` | Normalization scale factor |

### Batch Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `5000` | Samples per batch |
| `output_path` | `None` | Stream results to H5AD file |
| `output_compression` | `"gzip"` | Compression: "gzip", "lzf", or None |

## GPU Acceleration

```python
from secactpy import secact_activity_inference, CUPY_AVAILABLE

print(f"GPU available: {CUPY_AVAILABLE}")

# Auto-detect GPU
result = secact_activity_inference(expression, backend='auto')

# Force GPU
result = secact_activity_inference(expression, backend='cupy')
```

### Performance

| Dataset | R (Mac M1) | R (Linux) | Py (CPU) | Py (GPU) | Speedup |
|---------|------------|-----------|----------|----------|---------|
| Bulk (1,170 sp × 1,000 samples) | 74.4s | 141.6s | 128.8s | 6.7s | 11–19x |
| scRNA-seq (1,170 sp × 788 cells) | 54.9s | 117.4s | 104.8s | 6.8s | 8–15x |
| Visium (1,170 sp × 3,404 spots) | 141.7s | 379.8s | 381.4s | 11.2s | 13–34x |
| CosMx (151 sp × 443,515 cells) | 936.9s | 976.1s | 1226.7s | 99.9s | 9–12x |

<details>
<summary>Benchmark Environment</summary>

- **Mac CPU**: M1 Pro with VECLIB (8 cores)
- **Linux CPU**: AMD EPYC 7543P (4 cores)
- **Linux GPU**: NVIDIA A100-SXM4-80GB

</details>

## Command Line Interface

SecActPy provides a command line interface for common workflows:

```bash
# Bulk RNA-seq (differential expression)
secactpy bulk -i diff_expr.tsv -o results.h5ad --differential -v

# Bulk RNA-seq (raw counts)
secactpy bulk -i counts.tsv -o results.h5ad -v

# scRNA-seq with cell type aggregation
secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype -v

# scRNA-seq at single cell level
secactpy scrnaseq -i data.h5ad -o results.h5ad --single-cell -v

# Visium spatial transcriptomics
secactpy visium -i /path/to/visium/ -o results.h5ad -v

# CosMx (single-cell spatial)
secactpy cosmx -i cosmx.h5ad -o results.h5ad --batch-size 50000 -v

# Use GPU acceleration
secactpy bulk -i data.tsv -o results.h5ad --backend cupy -v

# Use CytoSig signature
secactpy bulk -i data.tsv -o results.h5ad --signature cytosig -v
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file or directory |
| `-o, --output` | Output H5AD file |
| `-s, --signature` | Signature matrix (secact, cytosig) |
| `--lambda` | Ridge regularization (default: 5e5) |
| `-n, --n-rand` | Number of permutations (default: 1000) |
| `--backend` | Computation backend (auto, numpy, cupy) |
| `--batch-size` | Batch size for large datasets |
| `-v, --verbose` | Verbose output |

## Docker

Pre-built Docker images are available:

```bash
# CPU version
docker pull psychemistz/secactpy:latest

# GPU version
docker pull psychemistz/secactpy:gpu

# With R SecAct/RidgeR for cross-validation
docker pull psychemistz/secactpy:with-r
```

See [DOCKER.md](DOCKER.md) for detailed usage instructions.

## Reproducibility

SecActPy produces **identical results** to R SecAct/RidgeR:

```python
result = secact_activity_inference(
    expression,
    is_differential=True,
    sig_matrix="secact",
    lambda_=5e5,
    n_rand=1000,
    seed=0,
    use_gsl_rng=True  # Default: R-compatible RNG
)
```

For faster analysis (when R compatibility is not required):

```python
result = secact_activity_inference(
    expression,
    use_gsl_rng=False,  # ~70x faster permutation generation
)
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- SciPy ≥ 1.7
- h5py ≥ 3.0
- anndata ≥ 0.8
- scanpy ≥ 1.9

**Optional:** CuPy ≥ 10.0 (GPU acceleration)

## Citation

If you use SecActPy in your research, please cite:

> Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang. 
> **Inference of secreted protein activities in intercellular communication.**
> [GitHub: data2intelligence/SecAct](https://github.com/data2intelligence/SecAct)

## Related Projects

- [SecAct](https://github.com/data2intelligence/SecAct) - Original R implementation
- [RidgeR](https://github.com/beibeiru/RidgeR) - R ridge regression package
- [SpaCET](https://github.com/data2intelligence/SpaCET) - Spatial transcriptomics cell type analysis
- [CytoSig](https://github.com/data2intelligence/CytoSig) - Cytokine signaling inference

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

### v0.2.0 (Official Release)
- Official release under data2intelligence organization
- PyPI package available (`pip install secactpy`)
- Comprehensive test suite and CI/CD pipeline
- Docker images with GPU and R support

### v0.1.2 (Initial Development)
- Ridge regression with permutation-based significance testing
- GPU acceleration via CuPy backend (9–34x speedup)
- Batch processing with streaming H5AD output for million-sample datasets
- Automatic sparse matrix handling in `ridge_batch()`
- Built-in SecAct and CytoSig signature matrices
- GSL-compatible RNG for R/RidgeR reproducibility
- Support for Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics
- Cell type resolution for ST data (`cell_type_col`, `is_spot_level`)
- Optional permutation table caching (`use_cache`)
