"""
SecActPy: Secreted Protein Activity Inference

A Python package for inferring secreted protein activity from
gene expression data using ridge regression with permutation testing.

Compatible with R's SecAct/RidgeR package - produces identical results.

Quick Start (Bulk RNA-seq):
---------------------------
    >>> from secactpy import secact_activity_inference
    >>>
    >>> # From file path (auto-detect format: CSV, TSV, TXT)
    >>> # Note: Set is_differential=True for log fold-change input.
    >>> # For single-column input with no control, row-mean centering
    >>> # is automatically skipped (it would produce all zeros).
    >>> result = secact_activity_inference(
    ...     "diff_expression.csv",  # or .tsv, .txt
    ...     is_differential=True,
    ...     verbose=True
    ... )
    >>>
    >>> # If genes are in first column (not row names)
    >>> result = secact_activity_inference(
    ...     "data.csv",
    ...     gene_col=0,  # genes in first column
    ...     is_differential=True
    ... )
    >>>
    >>> # Or from DataFrame
    >>> import pandas as pd
    >>> diff_expr = pd.read_csv("diff_expression.csv", index_col=0)
    >>> result = secact_activity_inference(diff_expr, is_differential=True)
    >>>
    >>> # Access results
    >>> activity = result['zscore']    # Activity z-scores
    >>> pvalues = result['pvalue']     # Significance

Flexible Data Loading:
----------------------
    >>> from secactpy import load_expression_data
    >>>
    >>> # Auto-detect format
    >>> expr = load_expression_data("data.csv")
    >>> expr = load_expression_data("data.tsv")
    >>> expr = load_expression_data("data.txt")
    >>>
    >>> # Genes in first column (not row names)
    >>> expr = load_expression_data("data.csv", gene_col=0)

scRNA-seq Analysis:
-------------------
    >>> import anndata as ad
    >>> from secactpy import secact_activity_inference_scrnaseq
    >>>
    >>> # Load AnnData (h5ad file)
    >>> adata = ad.read_h5ad("scrnaseq_data.h5ad")
    >>>
    >>> # Run pseudo-bulk analysis by cell type
    >>> result = secact_activity_inference_scrnaseq(
    ...     adata,
    ...     cell_type_col="cell_type",
    ...     is_single_cell_level=False,  # Aggregate by cell type
    ...     verbose=True
    ... )
    >>> activity = result['zscore']  # (proteins × cell_types)

Spatial Transcriptomics:
------------------------
    >>> from secactpy import secact_activity_inference_st, load_visium_10x
    >>>
    >>> # Load 10X Visium data
    >>> result = secact_activity_inference_st(
    ...     "path/to/visium_folder/",
    ...     min_genes=1000,
    ...     scale_factor=1e5,
    ...     verbose=True
    ... )
    >>> activity = result['zscore']  # (proteins × spots)

For large datasets (>100k samples):
-----------------------------------
    >>> from secactpy import ridge_batch, estimate_batch_size
    >>>
    >>> # Estimate optimal batch size
    >>> batch_size = estimate_batch_size(n_genes=20000, n_features=50)
    >>>
    >>> # Run batch processing with streaming output
    >>> ridge_batch(X, Y, batch_size=batch_size, output_path="results.h5ad")
    >>>
    >>> # Load results
    >>> from secactpy import load_results, results_to_anndata
    >>> results = load_results("results.h5ad")
    >>> adata = results_to_anndata(results)  # For scanpy integration
"""

__version__ = "0.2.3"

# Batch processing for large datasets
from .batch import (
    StreamingResultWriter,
    estimate_batch_size,
    estimate_memory,
    ridge_batch,
)

# High-level API (most users need only these)
from .inference import (
    compute_differential,
    expand_rows,
    group_signatures,
    load_expression_data,
    load_visium_10x,
    prepare_data,
    scale_columns,
    secact_activity,
    secact_activity_inference,
    secact_activity_inference_scrnaseq,
    secact_activity_inference_st,
)

# I/O utilities
from .io import (
    ANNDATA_AVAILABLE,
    H5PY_AVAILABLE,
    add_activity_to_anndata,
    concatenate_results,
    get_file_info,
    load_as_anndata,
    load_results,
    results_to_anndata,
    results_to_dataframes,
    save_results,
    save_st_results_to_h5ad,
)

# Lower-level ridge functions (for advanced users)
from .ridge import (
    CUPY_AVAILABLE,
    CUPY_INIT_ERROR,
    DEFAULT_LAMBDA,
    DEFAULT_NRAND,
    DEFAULT_SEED,
    compute_projection_matrix,
    ridge,
    ridge_with_precomputed_T,
)

# RNG (for reproducibility testing)
from .rng import (
    DEFAULT_CACHE_DIR,
    CStdlibRNG,
    GSLRNG,
    NumpyRNG,
    clear_perm_cache,
    generate_inverse_permutation_table,
    generate_inverse_permutation_table_fast,
    generate_permutation_table,
    generate_permutation_table_fast,
    get_cached_inverse_perm_table,
    get_cached_perm_table,
    list_cached_tables,
)

# Signature loading
from .signature import (
    AVAILABLE_SIGNATURES,
    get_signature_info,
    list_signatures,
    load_cytosig,
    load_lincytosig,
    load_secact,
    load_signature,
)

__all__ = [
    # Main API
    "secact_activity",
    "secact_activity_inference",
    "secact_activity_inference_scrnaseq",
    "secact_activity_inference_st",
    "load_visium_10x",
    "load_expression_data",
    "prepare_data",
    "scale_columns",
    "compute_differential",
    "group_signatures",
    "expand_rows",
    # Signature loading
    "load_signature",
    "load_secact",
    "load_cytosig",
    "load_lincytosig",
    "list_signatures",
    "get_signature_info",
    "AVAILABLE_SIGNATURES",
    # Ridge functions
    "ridge",
    "compute_projection_matrix",
    "ridge_with_precomputed_T",
    # Batch processing
    "ridge_batch",
    "estimate_batch_size",
    "estimate_memory",
    "StreamingResultWriter",
    "PopulationStats",
    "ProjectionComponents",
    "precompute_population_stats",
    "precompute_projection_components",
    "ridge_batch_sparse_preserving",
    # I/O
    "load_results",
    "save_results",
    "results_to_anndata",
    "load_as_anndata",
    "results_to_dataframes",
    "get_file_info",
    "concatenate_results",
    "save_st_results_to_h5ad",
    "add_activity_to_anndata",
    # RNG and Caching
    "CStdlibRNG",
    "GSLRNG",
    "NumpyRNG",
    "generate_permutation_table",
    "generate_inverse_permutation_table",
    "generate_permutation_table_fast",
    "generate_inverse_permutation_table_fast",
    "get_cached_perm_table",
    "get_cached_inverse_perm_table",
    "clear_perm_cache",
    "list_cached_tables",
    "DEFAULT_CACHE_DIR",
    # Availability flags
    "CUPY_AVAILABLE",
    "CUPY_INIT_ERROR",
    "H5PY_AVAILABLE",
    "ANNDATA_AVAILABLE",
    # Constants
    "DEFAULT_LAMBDA",
    "DEFAULT_NRAND",
    "DEFAULT_SEED",
    # Version
    "__version__",
]
