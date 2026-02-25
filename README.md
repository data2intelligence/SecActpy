# SecActPy

**Secreted Protein Activity Inference using Ridge Regression**

[![PyPI version](https://badge.fury.io/py/secactpy.svg)](https://pypi.org/project/secactpy/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/data2intelligence/SecActPy/actions/workflows/test.yml/badge.svg)](https://github.com/data2intelligence/SecActPy/actions/workflows/test.yml)
[![Docker](https://img.shields.io/docker/pulls/psychemistz/secactpy)](https://hub.docker.com/r/psychemistz/secactpy)

Python implementation of [SecAct](https://github.com/data2intelligence/SecAct) for inferring secreted protein activities from gene expression data.

**Key Features:**
- **SecAct Compatible**: Matches R SecAct/RidgeR results on the same platform (`rng_method='srand'`)
- **GPU Acceleration**: Optional CuPy backend for large-scale analysis
- **Million-Sample Scale**: Batch processing with streaming output for massive datasets
- **Built-in Signatures**: Includes SecAct and CytoSig signature matrices
- **Multi-Platform Support**: Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics (Visium, CosMx)
- **Smart Caching**: Optional permutation table caching for faster repeated analyses
- **Sparse-Aware**: Automatic memory-efficient processing for sparse single-cell data

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

# Single-cell resolution (one score per cell)
result = secact_activity_inference_st(
    adata,
    is_spot_level=True,         # Score each cell individually (default)
    batch_size=5000,            # Process in chunks to limit memory
    output_path="cosmx_sc_results.h5ad",  # Stream to disk
    verbose=True
)
# result is None when output_path is set; load with ad.read_h5ad()

# Cell-type resolution (pseudo-bulk by cell type)
result = secact_activity_inference_st(
    adata,
    cell_type_col="cellType",  # Column in adata.obs
    is_spot_level=False,        # Aggregate by cell type
    verbose=True
)

activity = result['zscore']  # (proteins × cell_types)
```

## Batch Processing

For large datasets (50,000+ samples), batch processing splits computation into
memory-efficient chunks while producing **mathematically identical** results.
The projection matrix is computed once, then samples are processed in chunks.
Set `batch_size` on any high-level function:

```python
result = secact_activity_inference(expr_df, ..., batch_size=5000)
result = secact_activity_inference_scrnaseq(adata, ..., batch_size=5000)
result = secact_activity_inference_st(adata, ..., batch_size=5000)
```

| Mode | Parameter | Return value | Memory for output |
|------|-----------|--------------|-------------------|
| In-memory (default) | `output_path=None` | `dict` of DataFrames | All results in RAM |
| Streaming | `output_path="results.h5ad"` | `None` | Only one batch at a time |

Setting `sparse_mode=True` keeps sparse Y matrices in sparse format end-to-end,
avoiding densification and reducing memory by orders of magnitude for highly
sparse single-cell data (<5% density: ~1.8x faster; results identical).

See [Batch Processing](docs/batch_processing.md) for worked examples and
streaming output details.

## API Reference

See [API Reference](docs/api_reference.md) for full function signatures, parameters, and options. For low-level `ridge()` / `ridge_batch()` usage, see [Advanced API](docs/advanced_api.md).

## GPU Acceleration

```python
from secactpy import secact_activity_inference, CUPY_AVAILABLE

print(f"GPU available: {CUPY_AVAILABLE}")
result = secact_activity_inference(expression, backend='auto')
```

| Dataset | Py (CPU) | Py (GPU) | Speedup |
|---------|----------|----------|---------|
| Bulk (1,170 sp × 1,000 samples) | 128.8s | 6.7s | 11–19x |
| scRNA-seq (1,170 sp × 788 cells) | 104.8s | 6.8s | 8–15x |
| Visium (1,170 sp × 3,404 spots) | 381.4s | 11.2s | 13–34x |
| CosMx (151 sp × 443,515 cells) | 1226.7s | 99.9s | 9–12x |

See [GPU Acceleration](docs/gpu_acceleration.md) for full benchmarks and CUDA setup.
See [DOCKER.md](DOCKER.md) for Docker vs native performance benchmarks.

## Command Line Interface

```bash
secactpy bulk -i diff_expr.tsv -o results.h5ad --differential -v
secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype -v
secactpy visium -i /path/to/visium/ -o results.h5ad -v
secactpy cosmx -i cosmx.h5ad -o results.h5ad --batch-size 50000 -v
```

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file or directory |
| `-o, --output` | Output H5AD file |
| `-s, --signature` | Signature matrix (secact, cytosig) |
| `--backend` | Computation backend (auto, numpy, cupy) |
| `--batch-size` | Batch size for large datasets |
| `-v, --verbose` | Verbose output |

See [CLI Reference](docs/cli.md) for all commands and options.

## Docker

```bash
docker pull psychemistz/secactpy:latest      # CPU
docker pull psychemistz/secactpy:gpu          # GPU
docker pull psychemistz/secactpy:with-r       # With R SecAct/RidgeR
```

See [DOCKER.md](DOCKER.md) for detailed usage instructions.

## Reproducibility

SecActPy supports three RNG backends for different reproducibility needs:

| `rng_method` | Description | Use case |
|---|---|---|
| `'srand'` | C stdlib `srand()`/`rand()` via ctypes | Match R SecAct/RidgeR results **on the same platform** |
| `'gsl'` | Mersenne Twister (GSL-compatible) | **Cross-platform** reproducibility within SecActPy |
| `'numpy'` | Native NumPy RNG (~70x faster) | Fast analysis when reproducibility with R is not needed |

```python
# Match R SecAct on same platform
result = secact_activity_inference(expr, rng_method="srand")

# Cross-platform reproducible (default)
result = secact_activity_inference(expr, rng_method="gsl")

# Fastest (~70x faster permutations)
result = secact_activity_inference(expr, rng_method="numpy")
```

See [Reproducibility](docs/reproducibility.md) for detailed examples.

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
> *Nature Methods*, 2026 (In press)

## Related Projects

- [SecAct](https://github.com/data2intelligence/SecAct) - Original R implementation
- [RidgeR](https://github.com/beibeiru/RidgeR) - R ridge regression package
- [SpaCET](https://github.com/data2intelligence/SpaCET) - Spatial transcriptomics cell type analysis
- [CytoSig](https://github.com/data2intelligence/CytoSig) - Cytokine signaling inference

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

### v0.2.4
- `col_center` and `col_scale` parameters for independent control of sparse in-flight normalization

### v0.2.3
- `rng_method` parameter for explicit RNG selection
- `is_group_sig=True` by default
