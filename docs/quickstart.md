# Quick Start

## Example Data

Example datasets for all tutorials are available on Zenodo:

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

## Example 1: Bulk RNA-seq

```python
import pandas as pd
from secactpy import secact_activity_inference

# Load differential expression data (genes x samples)
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

!!! note
    Set `is_differential=True` when the input is already log fold-change data.
    For single-column input with no control, row-mean centering is automatically skipped
    (it would produce all zeros).

## Example 2: scRNA-seq Analysis

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

## Example 3: Spatial Transcriptomics

### Visium (spot-level)

```python
from secactpy import secact_activity_inference_st

# Load Visium HCC data (3,415 spots)
# Download: https://zenodo.org/records/18520356/files/Visium_HCC_data.h5ad
result = secact_activity_inference_st(
    "Visium_HCC_data.h5ad",
    min_genes=1000,
    verbose=True
)

activity = result['zscore']  # (proteins x spots)
```

### CosMx (single-cell spatial)

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

activity = result['zscore']  # (proteins x cell_types)
```

## Command Line Interface

SecActPy also provides a CLI for common workflows:

```bash
secactpy bulk -i diff_expr.tsv -o results.h5ad --differential -v
secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype -v
secactpy visium -i /path/to/visium/ -o results.h5ad -v
secactpy cosmx -i cosmx.h5ad -o results.h5ad --batch-size 50000 -v
```

See [CLI Reference](cli.md) for all commands and options.
