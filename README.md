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

### From PyPI (Recommended)

```bash
pip install secactpy
```

### From GitHub

```bash
# CPU Only
pip install git+https://github.com/data2intelligence/SecActpy.git

# With GPU Support (CUDA 11.x)
pip install "secactpy[gpu] @ git+https://github.com/data2intelligence/SecActpy.git"
```

> **Note**: For CUDA 12.x, install CuPy separately: `pip install cupy-cuda12x`

### Development Installation

```bash
git clone https://github.com/data2intelligence/SecActpy.git
cd SecActpy
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage (Bulk RNA-seq)

```python
import pandas as pd
from secactpy import secact_activity_inference

# Load your differential expression data (genes × samples)
diff_expr = pd.read_csv("diff_expression.csv", index_col=0)

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

### Spatial Transcriptomics (10X Visium)

```python
from secactpy import secact_activity_inference_st

# Spot-level analysis
result = secact_activity_inference_st(
    "path/to/visium_folder/",
    min_genes=1000,
    verbose=True
)

activity = result['zscore']  # (proteins × spots)
```

### Spatial Transcriptomics with Cell Type Resolution

```python
import anndata as ad
from secactpy import secact_activity_inference_st

# Load annotated spatial data
adata = ad.read_h5ad("spatial_annotated.h5ad")

# Cell-type resolution (pseudo-bulk by cell type)
result = secact_activity_inference_st(
    adata,
    cell_type_col="cell_type",  # Column in adata.obs
    is_spot_level=False,        # Aggregate by cell type
    verbose=True
)

activity = result['zscore']  # (proteins × cell_types)
```

### scRNA-seq Analysis

```python
import anndata as ad
from secactpy import secact_activity_inference_scrnaseq

adata = ad.read_h5ad("scrnaseq_data.h5ad")

# Pseudo-bulk by cell type
result = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="cell_type",
    is_single_cell_level=False,
    verbose=True
)

# Single-cell level
result_sc = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="cell_type",
    is_single_cell_level=True,
    verbose=True
)
```

### Large-Scale Batch Processing

```python
from secactpy import ridge_batch

# Dense data (pre-scaled)
Y_scaled = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
result = ridge_batch(
    X, Y_scaled,
    batch_size=5000,
    n_rand=1000,
    backend='cupy',  # Use GPU
    verbose=True
)

# Sparse data (auto-scaled internally)
import scipy.sparse as sp
Y_sparse = sp.csr_matrix(counts)  # Raw counts
result = ridge_batch(
    X, Y_sparse,
    batch_size=10000,
    n_rand=1000,
    backend='auto',
    verbose=True
)

# Stream results to disk for very large datasets
ridge_batch(
    X, Y,
    batch_size=10000,
    output_path="results.h5ad",
    output_compression="gzip",
    verbose=True
)
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
