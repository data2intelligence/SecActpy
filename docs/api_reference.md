# API Reference

## High-Level Functions

| Function | Description |
|----------|-------------|
| `secact_activity_inference()` | Bulk RNA-seq inference |
| `secact_activity_inference_scrnaseq()` | scRNA-seq inference |
| `secact_activity_inference_st()` | Spatial transcriptomics inference |
| `load_signature(name='secact')` | Load built-in signature matrix |

## Core Functions

| Function | Description |
|----------|-------------|
| `ridge()` | Single-call ridge regression with permutation testing |
| `ridge_batch()` | Batch processing for large datasets (dense or sparse) |
| `estimate_batch_size()` | Estimate optimal batch size for available memory |
| `estimate_memory()` | Estimate memory requirements |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_matrix` | `"secact"` | Signature: "secact", "cytosig", or DataFrame |
| `lambda_` | `5e5` | Ridge regularization parameter |
| `n_rand` | `1000` | Number of permutations |
| `seed` | `0` | Random seed for reproducibility |
| `rng_method` | `None` | RNG backend: `'srand'` (match R, default), `'gsl'` (cross-platform), `'numpy'` (fast). `None` defaults to srand. |
| `is_group_sig` | `True` | Group similar signatures by correlation before regression |
| `backend` | `'auto'` | 'auto', 'numpy', or 'cupy' |
| `use_cache` | `False` | Cache permutation tables to disk |
| `sparse_mode` | `False` | Keep sparse Y in sparse format (avoids densification) |
| `col_center` | `True` | Subtract column means during sparse in-flight normalization |
| `col_scale` | `True` | Divide by column stds during sparse in-flight normalization |

## ST-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_type_col` | `None` | Column in AnnData.obs for cell type |
| `is_spot_level` | `True` | If False, aggregate by cell type |
| `scale_factor` | `1e5` | Normalization scale factor |

## Batch Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `None` | Samples per batch (`None` = all at once) |
| `output_path` | `None` | Stream results to H5AD file (requires `batch_size`) |
| `output_compression` | `"gzip"` | Compression: "gzip", "lzf", or None |

For low-level `ridge()` / `ridge_batch()` usage and sparse column normalization
details, see [Advanced API](advanced_api.md).
