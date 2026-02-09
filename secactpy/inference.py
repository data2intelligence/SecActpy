"""
High-level SecAct activity inference API.

This module provides the main user-facing functions for inferring
secreted protein activity from gene expression data.

Usage:
------
    >>> from secactpy import secact_activity
    >>>
    >>> # Basic usage with expression DataFrame
    >>> result = secact_activity(expression_df, signature_df)
    >>>
    >>> # Access results as DataFrames
    >>> activity = result['zscore']  # Activity z-scores
    >>> pvalues = result['pvalue']   # Significance

The main function `secact_activity()` handles:
- Gene overlap detection between expression and signature
- Z-score normalization of expression data
- Ridge regression with permutation testing
- Result formatting with proper row/column labels
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Optional, Union, Any, Literal
import time

from .ridge import ridge

__all__ = [
    'secact_activity',
    'secact_activity_inference',
    'secact_activity_inference_scrnaseq',
    'secact_activity_inference_st',
    'load_visium_10x',
    'load_expression_data',
    'prepare_data',
    'scale_columns',
    'compute_differential',
    'group_signatures',
    'expand_rows',
]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_expression_data(
    filepath: Union[str, Path],
    sep: Optional[str] = None,
    gene_col: Optional[Union[str, int]] = None,
    index_col: Optional[Union[str, int]] = None,
    header: Union[int, None] = 0,
    **kwargs
) -> pd.DataFrame:
    """
    Load expression data from various file formats.

    Handles multiple input formats:
    - Gene symbols as row names (index), columns as samples
    - First column as gene symbols, remaining columns as samples
    - TSV, CSV, or space-separated files (auto-detected)

    Parameters
    ----------
    filepath : str or Path
        Path to expression data file.
    sep : str, optional
        Column separator. If None, auto-detects from file extension:
        - .csv -> ','
        - .tsv, .txt -> '\\t' or whitespace
    gene_col : str or int, optional
        Column name or index containing gene symbols.
        If None, assumes genes are in the index (row names).
        If 0 or column name, uses that column as gene index.
    index_col : int, optional
        Column to use as row index when reading.
        Default: 0 if file appears to have row names, else None.
    header : int or None, default=0
        Row number to use as column names.
    **kwargs
        Additional arguments passed to pd.read_csv().

    Returns
    -------
    DataFrame
        Expression matrix with genes as rows (index) and samples as columns.

    Examples
    --------
    >>> # Auto-detect format
    >>> expr = load_expression_data("data.csv")
    >>> expr = load_expression_data("data.tsv")
    >>> expr = load_expression_data("data.txt")

    >>> # Genes in first column (not index)
    >>> expr = load_expression_data("data.csv", gene_col=0)

    >>> # Genes in named column
    >>> expr = load_expression_data("data.csv", gene_col="GeneSymbol")

    >>> # Custom separator
    >>> expr = load_expression_data("data.txt", sep="\\t")

    >>> # No header row
    >>> expr = load_expression_data("data.txt", header=None)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect separator from extension
    if sep is None:
        suffix = filepath.suffix.lower()
        if suffix == '.csv':
            sep = ','
        elif suffix in ['.tsv', '.tab']:
            sep = '\t'
        else:
            # Try to detect from first line
            sep = _detect_separator(filepath)

    # First pass: peek at file to determine structure
    if index_col is None and gene_col is None:
        index_col = _detect_index_col(filepath, sep, header)

    # Read the file
    try:
        if gene_col is not None:
            # Gene symbols in a specific column, not index
            df = pd.read_csv(filepath, sep=sep, header=header, index_col=None, **kwargs)

            # Set gene column as index
            if isinstance(gene_col, int):
                gene_col_name = df.columns[gene_col]
            else:
                gene_col_name = gene_col

            df = df.set_index(gene_col_name)
        else:
            # Gene symbols as row index
            df = pd.read_csv(filepath, sep=sep, header=header, index_col=index_col, **kwargs)
    except Exception as e:
        # Try with different separators
        for try_sep in ['\t', ',', r'\s+']:
            if try_sep == sep:
                continue
            try:
                df = pd.read_csv(filepath, sep=try_sep, header=header,
                                index_col=index_col if index_col is not None else 0, **kwargs)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not read file {filepath}: {e}")

    # Clean up index name
    if df.index.name is None:
        df.index.name = 'Gene'

    # Ensure numeric data
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    # Drop any non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(df.columns):
        dropped = set(df.columns) - set(numeric_cols)
        warnings.warn(f"Dropped non-numeric columns: {dropped}")
        df = df[numeric_cols]

    # Handle duplicate gene names
    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        warnings.warn(f"Found {n_dups} duplicate gene names. Keeping first occurrence.")
        df = df[~df.index.duplicated(keep='first')]

    return df


def _detect_separator(filepath: Path) -> str:
    """Detect the most likely separator in a file."""
    with open(filepath, 'r') as f:
        first_lines = [f.readline() for _ in range(3)]

    # Count potential separators
    text = ''.join(first_lines)
    tab_count = text.count('\t')
    comma_count = text.count(',')

    if tab_count > comma_count:
        return '\t'
    elif comma_count > 0:
        return ','
    else:
        return r'\s+'  # Whitespace


def _detect_index_col(filepath: Path, sep: str, header: int) -> Optional[int]:
    """Detect if the file has row names in the first column."""
    try:
        # Read just the header and first few rows
        df_peek = pd.read_csv(filepath, sep=sep, header=header, nrows=5, index_col=None)

        if df_peek.shape[1] < 2:
            return None

        first_col = df_peek.iloc[:, 0]

        # Check if first column looks like gene names (strings, not numbers)
        if first_col.dtype == object:
            # Check if values look like gene symbols
            try:
                pd.to_numeric(first_col)
                return None  # First column is numeric, not gene names
            except Exception:
                return 0  # First column is strings, likely gene names

        return None
    except Exception:
        return 0  # Default to first column as index


# =============================================================================
# Data Preparation Functions
# =============================================================================

def scale_columns(
    df: pd.DataFrame,
    method: Literal["zscore", "center", "none"] = "zscore",
    epsilon: float = 1e-12
) -> pd.DataFrame:
    """
    Scale DataFrame columns (samples).

    Parameters
    ----------
    df : DataFrame
        Data with genes as rows, samples as columns.
    method : {"zscore", "center", "none"}
        - "zscore": Standardize to mean=0, std=1 per column
        - "center": Center to mean=0 per column (no scaling)
        - "none": No transformation
    epsilon : float
        Small value added to std to prevent division by zero.

    Returns
    -------
    DataFrame
        Scaled data with same shape and labels.
    """
    if method == "none":
        return df.copy()

    values = df.values.astype(np.float64)

    # Column-wise centering
    means = np.nanmean(values, axis=0, keepdims=True)
    centered = values - means

    if method == "zscore":
        # Use ddof=1 to match R's scale() function which uses sample std (n-1 denominator)
        stds = np.nanstd(values, axis=0, keepdims=True, ddof=1)
        # Warn about near-zero std columns
        zero_std_mask = stds.ravel() < epsilon
        if zero_std_mask.any():
            n_zero = zero_std_mask.sum()
            warnings.warn(
                f"{n_zero} column(s) have near-zero variance. "
                "Z-scores for these will be 0.",
                RuntimeWarning
            )
        scaled = centered / (stds + epsilon)
    else:  # center
        scaled = centered

    # Handle NaN/Inf
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def prepare_data(
    expression: pd.DataFrame,
    signature: pd.DataFrame,
    scale: Literal["zscore", "center", "none"] = "zscore",
    min_genes: int = 10
) -> tuple:
    """
    Prepare expression and signature matrices for ridge regression.

    Handles:
    - Finding common genes between expression and signature
    - Aligning matrices to common gene order
    - Optional z-score scaling of expression data

    Parameters
    ----------
    expression : DataFrame, shape (n_genes, n_samples)
        Gene expression data. Rows are genes, columns are samples.
    signature : DataFrame, shape (n_genes, n_features)
        Signature matrix. Rows are genes, columns are proteins/features.
    scale : {"zscore", "center", "none"}
        How to scale expression columns.
    min_genes : int
        Minimum number of common genes required.

    Returns
    -------
    tuple
        (X, Y, feature_names, sample_names, gene_names)
        - X : ndarray (n_common_genes, n_features) - Signature matrix
        - Y : ndarray (n_common_genes, n_samples) - Expression matrix
        - feature_names : list - Column names from signature
        - sample_names : list - Column names from expression
        - gene_names : list - Common gene names

    Raises
    ------
    ValueError
        If fewer than min_genes common genes are found.
    """
    # Ensure string indices for matching
    expr_idx = expression.index.astype(str)
    sig_idx = signature.index.astype(str)

    # Find common genes
    common_genes = expr_idx.intersection(sig_idx)
    n_common = len(common_genes)

    if n_common < min_genes:
        raise ValueError(
            f"Only {n_common} common genes found between expression and signature. "
            f"Minimum required: {min_genes}. "
            "Check that gene identifiers match (e.g., both use gene symbols)."
        )

    if n_common < len(sig_idx):
        pct = 100 * n_common / len(sig_idx)
        warnings.warn(
            f"Using {n_common}/{len(sig_idx)} ({pct:.1f}%) signature genes. "
            f"Missing genes will reduce inference accuracy.",
            RuntimeWarning
        )

    # Align to common genes
    # Create temporary DataFrames with string indices
    expr_aligned = expression.copy()
    expr_aligned.index = expr_idx
    sig_aligned = signature.copy()
    sig_aligned.index = sig_idx

    # Subset to common genes (in signature order for reproducibility)
    common_genes_ordered = [g for g in sig_idx if g in common_genes]

    Y_df = expr_aligned.loc[common_genes_ordered]
    X_df = sig_aligned.loc[common_genes_ordered]

    # Scale expression data
    if scale != "none":
        Y_df = scale_columns(Y_df, method=scale)

    # Extract arrays
    X = X_df.values.astype(np.float64)
    Y = Y_df.values.astype(np.float64)

    # Store names for result labeling
    feature_names = list(X_df.columns)
    sample_names = list(Y_df.columns)
    gene_names = list(common_genes_ordered)

    return X, Y, feature_names, sample_names, gene_names


# =============================================================================
# Main Inference Function
# =============================================================================

def secact_activity(
    expression: pd.DataFrame,
    signature: pd.DataFrame,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    scale: Literal["zscore", "center", "none"] = "zscore",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    min_genes: int = 10,
    use_gsl_rng: bool = True,
    rng_method: Literal["srand", "gsl", "numpy", None] = None,
    use_cache: bool = False,
    batch_size: Optional[int] = None,
    sparse_mode: bool = False,
    verbose: bool = False
) -> dict[str, Any]:
    """
    Infer secreted protein activity from gene expression data.

    This is the main user-facing function that combines data preparation
    and ridge regression into a single convenient call.

    Parameters
    ----------
    expression : DataFrame, shape (n_genes, n_samples)
        Gene expression data. Rows are genes (e.g., gene symbols),
        columns are samples.
    signature : DataFrame, shape (n_genes, n_features)
        Signature matrix mapping genes to proteins/cytokines.
        Rows are genes, columns are protein names.
        Can be loaded from built-in signatures (see `load_cytosig()`).
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations for significance testing.
        Set to 0 for faster t-test based inference.
    seed : int, default=0
        Random seed for reproducibility.
        Use 0 for exact compatibility with RidgeR.
    scale : {"zscore", "center", "none"}, default="zscore"
        How to scale expression data before inference.
        - "zscore": Standardize each sample to mean=0, std=1
        - "center": Center each sample to mean=0
        - "none": No scaling
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Computation backend.
    min_genes : int, default=10
        Minimum number of overlapping genes required.
    use_gsl_rng : bool, default=True
        Use GSL-compatible RNG for exact R/RidgeR reproducibility.
        Set to False for faster inference (~70x faster permutation generation)
        when exact R matching is not needed.
    use_cache : bool, default=False
        Cache permutation tables to disk for reuse. Enable when running
        multiple analyses with the same gene count for faster repeated runs.
    sparse_mode : bool, default=False
        When True and Y is a sparse matrix, avoid densifying Y during
        computation. More memory-efficient for very sparse data (<5% density).
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    dict
        Results dictionary containing:

        - **beta** : DataFrame (n_features, n_samples)
            Regression coefficients (activity estimates).
        - **se** : DataFrame (n_features, n_samples)
            Standard errors.
        - **zscore** : DataFrame (n_features, n_samples)
            Z-scores (activity significance).
        - **pvalue** : DataFrame (n_features, n_samples)
            P-values from permutation test (or t-test if n_rand=0).
        - **n_genes** : int
            Number of genes used in inference.
        - **genes** : list
            Gene names used.
        - **method** : str
            Backend used ("numpy" or "cupy").
        - **time** : float
            Total execution time in seconds.

    Examples
    --------
    >>> import pandas as pd
    >>> from secactpy import secact_activity
    >>>
    >>> # Load your data
    >>> expression = pd.read_csv("expression.csv", index_col=0)
    >>> signature = pd.read_csv("signature.csv", index_col=0)
    >>>
    >>> # Run inference
    >>> result = secact_activity(expression, signature)
    >>>
    >>> # Get significant activities
    >>> significant = result['pvalue'] < 0.05
    >>> top_activities = result['zscore'][significant].stack().sort_values()

    >>> # Quick analysis with t-test (faster, less accurate)
    >>> result_fast = secact_activity(expression, signature, n_rand=0)

    Notes
    -----
    The function performs the following steps:

    1. Find common genes between expression and signature
    2. Align matrices to common gene order
    3. Optionally z-score normalize expression columns
    4. Run ridge regression: β = (X'X + λI)^{-1} X' Y
    5. Compute significance via permutation testing
    6. Return results as labeled DataFrames

    For compatibility with R's SecAct/RidgeR package, use the default
    parameters (lambda_=5e5, n_rand=1000, seed=0, scale="zscore").
    """
    start_time = time.time()

    # --- Input Validation ---
    if not isinstance(expression, pd.DataFrame):
        raise TypeError(
            f"expression must be a pandas DataFrame, got {type(expression).__name__}"
        )
    if not isinstance(signature, pd.DataFrame):
        raise TypeError(
            f"signature must be a pandas DataFrame, got {type(signature).__name__}"
        )

    if verbose:
        print("SecAct Activity Inference")
        print(f"  Expression: {expression.shape[0]} genes × {expression.shape[1]} samples")
        print(f"  Signature: {signature.shape[0]} genes × {signature.shape[1]} features")

    # --- Prepare Data ---
    if verbose:
        print("  Preparing data...")

    X, Y, feature_names, sample_names, gene_names = prepare_data(
        expression=expression,
        signature=signature,
        scale=scale,
        min_genes=min_genes
    )

    n_genes_used = len(gene_names)
    if verbose:
        print(f"  Using {n_genes_used} common genes (of {signature.shape[0]} signature genes)")

    # --- Run Ridge Regression ---
    if verbose:
        print(f"  Running ridge regression (n_rand={n_rand})...")

    # Use batch processing if batch_size is specified
    if batch_size is not None:
        from .batch import ridge_batch
        ridge_result = ridge_batch(
            X=X,
            Y=Y,
            lambda_=lambda_,
            n_rand=n_rand,
            seed=seed,
            batch_size=batch_size,
            backend=backend,
            use_gsl_rng=use_gsl_rng,
            rng_method=rng_method,
            use_cache=use_cache,
            sparse_mode=sparse_mode,
            verbose=verbose
        )
    else:
        ridge_result = ridge(
            X=X,
            Y=Y,
            lambda_=lambda_,
            n_rand=n_rand,
            seed=seed,
            backend=backend,
            use_gsl_rng=use_gsl_rng,
            rng_method=rng_method,
            use_cache=use_cache,
            sparse_mode=sparse_mode,
            verbose=verbose
        )

    # --- Format Results as DataFrames ---
    if verbose:
        print("  Formatting results...")

    beta_df = pd.DataFrame(
        ridge_result['beta'],
        index=feature_names,
        columns=sample_names
    )
    se_df = pd.DataFrame(
        ridge_result['se'],
        index=feature_names,
        columns=sample_names
    )
    zscore_df = pd.DataFrame(
        ridge_result['zscore'],
        index=feature_names,
        columns=sample_names
    )
    pvalue_df = pd.DataFrame(
        ridge_result['pvalue'],
        index=feature_names,
        columns=sample_names
    )

    total_time = time.time() - start_time

    if verbose:
        print(f"  Completed in {total_time:.2f}s")

    # --- Build Result Dictionary ---
    result = {
        # Main results as DataFrames
        'beta': beta_df,
        'se': se_df,
        'zscore': zscore_df,
        'pvalue': pvalue_df,

        # Metadata
        'n_genes': n_genes_used,
        'genes': gene_names,
        'features': feature_names,
        'samples': sample_names,

        # Execution info
        'method': ridge_result['method'],
        'time': total_time,
        'ridge_time': ridge_result['time'],

        # Parameters used
        'params': {
            'lambda_': lambda_,
            'n_rand': n_rand,
            'seed': seed,
            'scale': scale,
            'backend': backend
        }
    }

    # Add t-test df if applicable
    if 'df' in ridge_result:
        result['df'] = ridge_result['df']

    return result


# =============================================================================
# Differential Expression Helper
# =============================================================================

def compute_differential(
    treatment: pd.DataFrame,
    control: Optional[pd.DataFrame] = None,
    paired: bool = False,
    aggregate: bool = True
) -> pd.DataFrame:
    """
    Compute differential expression profile.

    Parameters
    ----------
    treatment : DataFrame
        Treatment expression (genes × samples).
    control : DataFrame, optional
        Control expression (genes × samples).
        If None, centers treatment by row means.
    paired : bool, default=False
        If True and control provided, compute paired differences
        (requires matching column names).
    aggregate : bool, default=True
        If True, average across samples to get single profile.

    Returns
    -------
    DataFrame
        Differential expression profile.
    """
    treatment = treatment.copy()
    treatment.index = treatment.index.astype(str)

    if control is None:
        # Center by row means
        row_means = treatment.mean(axis=1)
        diff = treatment.subtract(row_means, axis=0)
    else:
        control = control.copy()
        control.index = control.index.astype(str)

        # Find common genes
        common_genes = treatment.index.intersection(control.index)
        if len(common_genes) == 0:
            raise ValueError("No common genes between treatment and control")

        treatment = treatment.loc[common_genes]
        control = control.loc[common_genes]

        if paired:
            # Paired differences (matching samples)
            common_samples = treatment.columns.intersection(control.columns)
            if len(common_samples) == 0:
                raise ValueError("No matching sample names for paired analysis")
            diff = treatment[common_samples] - control[common_samples]
        else:
            # Difference from control mean
            control_mean = control.mean(axis=1)
            diff = treatment.subtract(control_mean, axis=0)

    if aggregate:
        diff = pd.DataFrame({'differential': diff.mean(axis=1)})

    # Handle NaN
    diff = diff.fillna(0)

    return diff


# =============================================================================
# Signature Grouping (matching R's SecAct.activity.inference)
# =============================================================================

def group_signatures(
    X: pd.DataFrame,
    cor_threshold: float = 0.9
) -> pd.DataFrame:
    """
    Group similar signatures by Pearson correlation.

    Matches R's .group_signatures function from RidgeR:
    - Calculate correlation-based distance
    - Hierarchical clustering (complete linkage)
    - Cut tree at 1 - cor_threshold
    - Average signatures within groups
    - Name groups as "A|B|C"

    Parameters
    ----------
    X : DataFrame
        Signature matrix (genes × proteins)
    cor_threshold : float, default=0.9
        Correlation threshold for grouping.
        Signatures with correlation >= threshold are grouped together.

    Returns
    -------
    DataFrame
        Grouped signature matrix with averaged values and pipe-delimited names.

    Examples
    --------
    >>> sig = load_signature('secact')
    >>> grouped = group_signatures(sig, cor_threshold=0.9)
    >>> print(f"Reduced from {sig.shape[1]} to {grouped.shape[1]} groups")
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    n_features = X.shape[1]

    # Handle edge case: single column
    if n_features <= 1:
        return X.copy()

    # Calculate correlation distance (1 - correlation)
    # pdist expects samples as rows, so we transpose X
    # metric='correlation' computes 1 - pearson_correlation
    try:
        dist_condensed = pdist(X.T.values, metric='correlation')
        # Handle NaN in distances (from constant columns)
        dist_condensed = np.nan_to_num(dist_condensed, nan=1.0)
    except Exception:
        # Fallback: compute manually
        corr_matrix = X.corr(method='pearson')
        corr_matrix = corr_matrix.fillna(0).clip(-1, 1)
        n = len(corr_matrix)
        dist_condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_condensed.append(1 - corr_matrix.iloc[i, j])
        dist_condensed = np.array(dist_condensed)

    # Hierarchical clustering with complete linkage (matching R)
    Z = linkage(dist_condensed, method='complete')

    # Cut tree at distance = 1 - cor_threshold
    cut_height = 1 - cor_threshold
    group_labels = fcluster(Z, t=cut_height, criterion='distance')

    # Create mapping from protein to group
    protein_names = list(X.columns)
    protein_groups = dict(zip(protein_names, group_labels))

    # Build new signature matrix with grouped signatures
    # Use list to collect columns, then concat at once (avoiding fragmentation warning)
    group_data = {}

    for group_id in sorted(set(group_labels)):
        # Get proteins in this group (sorted for consistent naming)
        proteins_in_group = sorted([p for p, g in protein_groups.items() if g == group_id])

        # Create group name (e.g., "A|B|C")
        group_name = "|".join(proteins_in_group)

        # Average signatures within group
        group_data[group_name] = X[proteins_in_group].mean(axis=1)

    new_sig = pd.DataFrame(group_data, index=X.index)

    return new_sig


def expand_rows(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Expand rows with pipe-delimited names.

    Matches R's .expand_rows function from RidgeR:
    Expands rows where index contains "|" (grouped signatures) into
    separate rows with duplicated values.

    Parameters
    ----------
    mat : DataFrame
        Matrix with potentially grouped row names (e.g., "A|B|C")

    Returns
    -------
    DataFrame
        Matrix with expanded rows

    Examples
    --------
    >>> # If mat has row "IL6|IL6R" with values [1, 2]
    >>> # Result will have two rows: "IL6" with [1, 2] and "IL6R" with [1, 2]
    >>> expanded = expand_rows(mat)
    """
    new_rows = []
    new_names = []

    for idx in mat.index:
        idx_str = str(idx)
        if "|" in idx_str:
            # Split the grouped name
            split_names = idx_str.split("|")
            for name in split_names:
                new_rows.append(mat.loc[idx].values)
                new_names.append(name)
        else:
            new_rows.append(mat.loc[idx].values)
            new_names.append(idx_str)

    result = pd.DataFrame(new_rows, index=new_names, columns=mat.columns)
    return result


# =============================================================================
# Full Inference Function (matching R's SecAct.activity.inference)
# =============================================================================

def secact_activity_inference(
    input_profile: Union[pd.DataFrame, str, Path],
    input_profile_control: Union[pd.DataFrame, str, Path, None] = None,
    is_differential: bool = False,
    is_paired: bool = False,
    is_single_sample_level: bool = False,
    sig_matrix: str = "secact",
    is_group_sig: bool = True,
    is_group_cor: float = 0.9,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    sig_filter: bool = False,
    gene_col: Optional[Union[str, int]] = None,
    backend: str = "numpy",
    use_gsl_rng: bool = True,
    rng_method: Literal["srand", "gsl", "numpy", None] = None,
    use_cache: bool = False,
    batch_size: Optional[int] = None,
    output_path: Optional[str] = None,
    output_compression: Optional[str] = "gzip",
    sort_genes: bool = False,
    sparse_mode: bool = False,
    verbose: bool = True
) -> Optional[dict[str, pd.DataFrame]]:
    """
    Secreted Protein Activity Inference (matching R's SecAct.activity.inference).

    This function provides full compatibility with the R SecAct package,
    including signature grouping and row expansion.

    Parameters
    ----------
    input_profile : DataFrame, str, or Path
        Gene expression matrix (genes × samples). Can be:
        - pandas DataFrame with genes as index
        - Path to file (CSV, TSV, or TXT)

        Supported file formats:
        - CSV (comma-separated)
        - TSV (tab-separated)
        - TXT (space or tab-separated)

        File can have genes as:
        - Row names (index): first column is gene symbols
        - First column: use gene_col=0 parameter

    input_profile_control : DataFrame, str, Path, or None, optional
        Control expression matrix (genes × samples).
        Accepts same formats as input_profile.
        If None and is_differential=False, uses mean of input_profile as control.
    is_differential : bool, default=False
        If True, input_profile is already differential expression (log fold-change).
        Note: single-column input with no control automatically skips
        row-mean centering (since it would produce all zeros).
    is_paired : bool, default=False
        If True, perform paired differential calculation.
    is_single_sample_level : bool, default=False
        If True, calculate per-sample activity.
    sig_matrix : str or DataFrame, default="secact"
        Signature matrix. Either "secact", "cytosig", path to file, or DataFrame.
    is_group_sig : bool, default=True
        Whether to group similar signatures by correlation.
    is_group_cor : float, default=0.9
        Correlation threshold for grouping.
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations.
    seed : int, default=0
        Random seed for reproducibility.
    sig_filter : bool, default=False
        If True, filter signatures by available genes.
    gene_col : str or int, optional
        Column containing gene symbols (if not row index).
        Use gene_col=0 if genes are in the first column.
    backend : str, default="numpy"
        Computation backend ("numpy" or "cupy").
    use_gsl_rng : bool, default=True
        Use GSL-compatible RNG for exact R/RidgeR reproducibility.
        Set to False for faster inference when R matching is not needed.
    use_cache : bool, default=False
        Cache permutation tables to disk for reuse. Enable when running
        multiple analyses with the same gene count for faster repeated runs.
    output_path : str, optional
        Path to an HDF5 file for streaming results to disk. When set,
        each batch's results are written directly to this file as they
        complete, and the function returns ``None`` instead of a dict.
        Requires ``batch_size`` to be set.
    output_compression : str, optional, default="gzip"
        Compression for the HDF5 output file. One of "gzip", "lzf",
        or None. Only used when ``output_path`` is set.
    sort_genes : bool, default=False
        If True, sort common genes alphabetically before running ridge regression.
        This ensures reproducible results across different platforms but may
        differ from original gene order. Set to True for cross-platform
        reproducibility with R.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    dict or None
        If ``output_path`` is None (default), returns a results dictionary
        containing:

        - beta : DataFrame (proteins × samples) - Regression coefficients
        - se : DataFrame (proteins × samples) - Standard errors
        - zscore : DataFrame (proteins × samples) - Z-scores
        - pvalue : DataFrame (proteins × samples) - P-values

        If ``output_path`` is set, results are streamed to the file and
        the function returns ``None``.

    Raises
    ------
    ValueError
        If ``output_path`` is set without ``batch_size``.

    Examples
    --------
    >>> # From file path (auto-detect format)
    >>> result = secact_activity_inference("diff_expr.csv", is_differential=True)
    >>> result = secact_activity_inference("diff_expr.tsv", is_differential=True)
    >>> result = secact_activity_inference("diff_expr.txt", is_differential=True)

    >>> # From file with genes in first column
    >>> result = secact_activity_inference("data.csv", gene_col=0, is_differential=True)

    >>> # From DataFrame
    >>> expr_diff = pd.read_csv("Ly86-Fc_vs_Vehicle_logFC.txt", sep="\\t", index_col=0)
    >>> result = secact_activity_inference(expr_diff, is_differential=True)

    >>> # Access results
    >>> print(result['zscore'].head())

    >>> # Stream results to disk for large datasets
    >>> secact_activity_inference(
    ...     large_expression_df,
    ...     batch_size=5000,
    ...     output_path="results.h5ad",
    ...     verbose=True
    ... )
    """
    from .signature import load_signature

    # --- Step 0: Load input data if file path ---
    if isinstance(input_profile, (str, Path)):
        if verbose:
            print(f"  Loading input file: {input_profile}")
        input_profile = load_expression_data(input_profile, gene_col=gene_col)
        if verbose:
            print(f"  Loaded: {input_profile.shape[0]} genes × {input_profile.shape[1]} samples")

    if input_profile_control is not None and isinstance(input_profile_control, (str, Path)):
        if verbose:
            print(f"  Loading control file: {input_profile_control}")
        input_profile_control = load_expression_data(input_profile_control, gene_col=gene_col)
        if verbose:
            print(f"  Loaded control: {input_profile_control.shape[0]} genes × {input_profile_control.shape[1]} samples")

    # Ensure input_profile is a DataFrame
    if not isinstance(input_profile, pd.DataFrame):
        raise ValueError("input_profile must be a DataFrame or path to expression file")

    # Validate streaming output requirements
    if output_path is not None and batch_size is None:
        raise ValueError(
            "output_path requires batch_size. "
            "Set batch_size (e.g., batch_size=5000) to enable streaming output."
        )

    # --- Step 1: Load signature matrix ---
    if isinstance(sig_matrix, pd.DataFrame):
        X = sig_matrix.copy()
    elif isinstance(sig_matrix, str):
        if sig_matrix.lower() in ["secact", "cytosig"]:
            X = load_signature(sig_matrix)
        else:
            # Assume it's a file path
            X = pd.read_csv(sig_matrix, sep='\t', index_col=0)
    else:
        raise ValueError("sig_matrix must be 'secact', 'cytosig', a file path, or a DataFrame")

    if verbose:
        print(f"  Loaded signature: {X.shape[0]} genes × {X.shape[1]} proteins")

    # --- Step 2: Compute differential expression if needed ---
    if is_differential:
        Y = input_profile.copy()
        if Y.shape[1] == 1 and Y.columns[0] != "Change":
            Y.columns = ["Change"]
    else:
        if input_profile_control is None:
            if input_profile.shape[1] == 1:
                # Single column: centering by row means would produce all
                # zeros because each row's mean equals its own value.
                # Skip centering and use the column as-is.
                Y = input_profile.copy()
                if Y.columns[0] != "Change":
                    Y.columns = ["Change"]
            else:
                # Center by row means
                row_means = input_profile.mean(axis=1)
                Y = input_profile.subtract(row_means, axis=0)
        else:
            if is_paired:
                # Paired differences
                common_samples = input_profile.columns.intersection(input_profile_control.columns)
                Y = input_profile[common_samples] - input_profile_control[common_samples]
            else:
                # Difference from control mean
                control_mean = input_profile_control.mean(axis=1)
                Y = input_profile.subtract(control_mean, axis=0)

            if not is_single_sample_level:
                # Aggregate to single column
                Y = pd.DataFrame({"Change": Y.mean(axis=1)})

    # --- Step 3: Filter signatures if requested ---
    # sig_filter keeps only proteins whose names also appear as genes in expression data
    # This is useful for panels like CosMx where only ~1000 genes are measured
    if sig_filter:
        n_before = X.shape[1]
        available_genes = set(Y.index)
        X = X.loc[:, X.columns.isin(available_genes)]
        if verbose:
            print(f"  sig_filter: kept {X.shape[1]} / {n_before} proteins")

    # --- Step 4: Group similar signatures if requested ---
    if is_group_sig:
        if verbose:
            print(f"  Grouping signatures (cor_threshold={is_group_cor})...")
        X = group_signatures(X, cor_threshold=is_group_cor)
        if verbose:
            print(f"  Grouped into {X.shape[1]} signature groups")

    # --- Step 5: Find overlapping genes ---
    common_genes = Y.index.intersection(X.index)

    # Optionally sort genes alphabetically
    if sort_genes:
        common_genes = common_genes.sort_values()
        if verbose:
            print("  Sorted genes alphabetically")

    if verbose:
        print(f"  Common genes: {len(common_genes)}")

    if len(common_genes) < 2:
        raise ValueError(
            f"Too few overlapping genes ({len(common_genes)}) between expression and signature matrices! "
            "Check that gene identifiers match (e.g., both use gene symbols)."
        )

    # --- Step 6: Subset to common genes ---
    X_aligned = X.loc[common_genes].astype(np.float64)
    Y_aligned = Y.loc[common_genes].astype(np.float64)

    # --- Step 7: Scale (z-score normalize columns) ---
    # R's scale() function: (x - mean) / sd where sd uses n-1 denominator (ddof=1)
    X_scaled = (X_aligned - X_aligned.mean()) / X_aligned.std(ddof=1)
    Y_scaled = (Y_aligned - Y_aligned.mean()) / Y_aligned.std(ddof=1)

    # --- Step 8: Replace NaN with 0 (from constant columns) ---
    X_scaled = X_scaled.fillna(0)
    Y_scaled = Y_scaled.fillna(0)

    # --- Step 9: Run ridge regression ---
    if verbose:
        print(f"  Running ridge regression (n_rand={n_rand})...")

    # Use batch processing if batch_size is specified
    if batch_size is not None:
        from .batch import ridge_batch
        result = ridge_batch(
            X=X_scaled.values,
            Y=Y_scaled.values,
            lambda_=lambda_,
            n_rand=n_rand,
            seed=seed,
            batch_size=batch_size,
            backend=backend,
            use_gsl_rng=use_gsl_rng,
            rng_method=rng_method,
            use_cache=use_cache,
            output_path=output_path,
            output_compression=output_compression,
            feature_names=X_scaled.columns.tolist(),
            sample_names=Y_scaled.columns.tolist(),
            sparse_mode=sparse_mode,
            verbose=False
        )
    else:
        result = ridge(
            X=X_scaled.values,
            Y=Y_scaled.values,
            lambda_=lambda_,
            n_rand=n_rand,
            seed=seed,
            backend=backend,
            use_gsl_rng=use_gsl_rng,
            rng_method=rng_method,
            use_cache=use_cache,
            sparse_mode=sparse_mode,
            verbose=False
        )

    # --- Step 10: Handle streaming output ---
    if result is None:
        # Streaming mode: results written to output_path by ridge_batch
        if verbose:
            print(f"  Results streamed to {output_path}")
        return None

    # --- Step 11: Create DataFrames with proper labels ---
    feature_names = X_scaled.columns.tolist()
    sample_names = Y_scaled.columns.tolist()

    beta_df = pd.DataFrame(result['beta'], index=feature_names, columns=sample_names)
    se_df = pd.DataFrame(result['se'], index=feature_names, columns=sample_names)
    zscore_df = pd.DataFrame(result['zscore'], index=feature_names, columns=sample_names)
    pvalue_df = pd.DataFrame(result['pvalue'], index=feature_names, columns=sample_names)

    # --- Step 12: Expand grouped signatures back to individual rows ---
    if is_group_sig:
        if verbose:
            print("  Expanding grouped signatures...")
        beta_df = expand_rows(beta_df)
        se_df = expand_rows(se_df)
        zscore_df = expand_rows(zscore_df)
        pvalue_df = expand_rows(pvalue_df)

        # Sort by row name (matching R's behavior)
        row_order = sorted(beta_df.index)
        beta_df = beta_df.loc[row_order]
        se_df = se_df.loc[row_order]
        zscore_df = zscore_df.loc[row_order]
        pvalue_df = pvalue_df.loc[row_order]

    if verbose:
        print(f"  Result shape: {beta_df.shape}")

    return {
        'beta': beta_df,
        'se': se_df,
        'zscore': zscore_df,
        'pvalue': pvalue_df
    }


# =============================================================================
# Internal: Signature preparation for sparse pipeline
# =============================================================================

def _prepare_signature_for_sparse(
    gene_names: list,
    sig_matrix,
    is_group_sig: bool,
    is_group_cor: float,
    sig_filter: bool,
    sort_genes: bool,
    verbose: bool
) -> tuple:
    """
    Load, group, and align signature matrix for sparse Y pipeline.

    Returns (X_scaled, common_genes, is_group_sig_flag) where X_scaled is
    the z-scored signature DataFrame aligned to common genes.
    """
    from .signature import load_signature

    # Load signature
    if isinstance(sig_matrix, pd.DataFrame):
        X = sig_matrix.copy()
    elif isinstance(sig_matrix, str):
        if sig_matrix.lower() in ["secact", "cytosig"]:
            X = load_signature(sig_matrix)
        else:
            X = pd.read_csv(sig_matrix, sep='\t', index_col=0)
    else:
        raise ValueError("sig_matrix must be 'secact', 'cytosig', a file path, or a DataFrame")

    if verbose:
        print(f"  Loaded signature: {X.shape[0]} genes × {X.shape[1]} proteins")

    # Filter signatures by available genes
    gene_set = set(gene_names)
    if sig_filter:
        n_before = X.shape[1]
        X = X.loc[:, X.columns.isin(gene_set)]
        if verbose:
            print(f"  sig_filter: kept {X.shape[1]} / {n_before} proteins")

    # Group similar signatures
    if is_group_sig:
        if verbose:
            print(f"  Grouping signatures (cor_threshold={is_group_cor})...")
        X = group_signatures(X, cor_threshold=is_group_cor)
        if verbose:
            print(f"  Grouped into {X.shape[1]} signature groups")

    # Find common genes (preserve signature order)
    common_genes = [g for g in X.index if g in gene_set]

    if sort_genes:
        common_genes = sorted(common_genes)

    if verbose:
        print(f"  Common genes: {len(common_genes)}")

    if len(common_genes) < 2:
        raise ValueError(
            f"Too few overlapping genes ({len(common_genes)}) between expression and signature. "
            "Check that gene identifiers match (e.g., both use gene symbols)."
        )

    # Align and scale X only (Y stays sparse)
    X_aligned = X.loc[common_genes].astype(np.float64)
    X_scaled = (X_aligned - X_aligned.mean()) / X_aligned.std(ddof=1)
    X_scaled = X_scaled.fillna(0)

    return X_scaled, common_genes


# =============================================================================
# scRNAseq Activity Inference (matching R's SecAct.activity.inference.scRNAseq)
# =============================================================================

def secact_activity_inference_scrnaseq(
    adata: Union[Any, str, Path],
    cell_type_col: str,
    sig_matrix: str = "secact",
    is_single_cell_level: bool = False,
    is_group_sig: bool = True,
    is_group_cor: float = 0.9,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    sig_filter: bool = False,
    backend: str = "auto",
    use_gsl_rng: bool = True,
    rng_method: Literal["srand", "gsl", "numpy", None] = None,
    use_cache: bool = False,
    batch_size: Optional[int] = None,
    output_path: Optional[str] = None,
    output_compression: Optional[str] = "gzip",
    sort_genes: bool = False,
    sparse_mode: bool = False,
    verbose: bool = False
) -> Optional[dict[str, Any]]:
    """
    Cell State Activity Inference from Single Cell RNA-seq Data.

    Calculate secreted protein signaling activity from scRNA-seq data.
    Matches R's RidgeR::SecAct.activity.inference.scRNAseq behavior.

    Parameters
    ----------
    adata : AnnData or str
        AnnData object with raw counts in adata.raw.X or adata.X,
        or path to .h5ad file.
        Gene names in adata.var_names or adata.raw.var_names.
    cell_type_col : str
        Column name in adata.obs containing cell type annotations.
    sig_matrix : str, default="secact"
        Signature matrix: "secact", "cytosig", or path to custom file.
    is_single_cell_level : bool, default=False
        If True, calculate activity for each single cell.
        If False, aggregate to pseudo-bulk by cell type.
    is_group_sig : bool, default=True
        If True, group similar signatures by correlation.
    is_group_cor : float, default=0.9
        Correlation threshold for signature grouping.
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations for significance testing.
    seed : int, default=0
        Random seed for reproducibility.
    sig_filter : bool, default=False
        If True, filter signatures by available genes.
    backend : str, default="auto"
        Computation backend: "auto", "numpy", "cupy".
    use_gsl_rng : bool, default=True
        Use GSL-compatible RNG for exact R/RidgeR reproducibility.
        Set to False for faster inference when R matching is not needed.
    use_cache : bool, default=False
        Cache permutation tables to disk for reuse.
    output_path : str, optional
        Path to an HDF5 file for streaming results to disk. When set,
        results are written as each batch completes and the function
        returns ``None``. Requires ``batch_size``.
    output_compression : str, optional, default="gzip"
        Compression for the HDF5 output file ("gzip", "lzf", or None).
    sort_genes : bool, default=False
        If True, sort common genes alphabetically before running ridge regression.
        This ensures reproducible results across different platforms.
    verbose : bool, default=False
        If True, print progress messages.

    Returns
    -------
    dict or None
        If ``output_path`` is None, returns a dictionary with:

        - 'beta': DataFrame of coefficients (proteins × cell_types/cells)
        - 'se': DataFrame of standard errors
        - 'zscore': DataFrame of z-scores
        - 'pvalue': DataFrame of p-values

        If ``output_path`` is set, returns ``None``.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> result = secact_activity_inference_scrnaseq(
    ...     adata,
    ...     cell_type_col="cell_type",
    ...     is_single_cell_level=False
    ... )
    >>> activity = result['zscore']
    """
    import importlib.util
    if importlib.util.find_spec("anndata") is None:
        raise ImportError(
            "anndata is required for scRNAseq analysis. "
            "Install with: pip install anndata"
        )
    if importlib.util.find_spec("scipy") is None:
        raise ImportError(
            "scipy is required for scRNAseq analysis. "
            "Install with: pip install scipy"
        )
    import anndata as ad
    from scipy import sparse

    if verbose:
        print("SecActPy scRNAseq Activity Inference")
        print("=" * 50)

    # --- Step 0: Load AnnData if path is provided ---
    if isinstance(adata, (str, Path)):
        adata_path = str(adata)
        if verbose:
            print(f"  Loading: {adata_path}")
        adata = ad.read_h5ad(adata_path)
        if verbose:
            print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # --- Step 1: Extract count matrix ---
    # AnnData stores as (cells × genes), we need (genes × cells) like R
    # Prefer raw counts if available
    
    # Determine which data layer to use
    use_layer = None
    
    if adata.raw is not None:
        counts_raw = adata.raw.X
        gene_names = list(adata.raw.var_names)
        if verbose:
            print(f"  Using raw counts: {counts_raw.shape} (cells × genes)")
    else:
        counts_raw = adata.X
        gene_names = list(adata.var_names)
        
        # Check if data looks like counts or is already normalized
        if hasattr(counts_raw, 'toarray'):
            sample_data = counts_raw[:min(1000, counts_raw.shape[0]), :min(100, counts_raw.shape[1])].toarray()
        else:
            sample_data = counts_raw[:min(1000, counts_raw.shape[0]), :min(100, counts_raw.shape[1])]
        
        max_val = np.max(sample_data)
        is_integer = np.allclose(sample_data[sample_data != 0], 
                                  np.round(sample_data[sample_data != 0]))
        
        # Data with max < 20 and non-integer values is likely log-normalized
        is_likely_normalized = max_val < 20 and not is_integer
        
        if verbose:
            print(f"  Using adata.X: {counts_raw.shape} (cells × genes)")
            if is_likely_normalized:
                print(f"  WARNING: Data appears to be already normalized (max={max_val:.2f})")
                print(f"           For accurate results, provide raw counts in adata.raw")
                print(f"           Attempting to proceed with normalized data...")

    # Transpose to (genes × cells) to match R convention
    if sparse.issparse(counts_raw):
        counts = counts_raw.T.tocsr()  # Now (genes × cells)
    else:
        counts = counts_raw.T  # Now (genes × cells)

    if verbose:
        print(f"  Transposed to: {counts.shape} (genes × cells)")

    # Get cell names and cell types
    cell_names = list(adata.obs_names)

    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs. "
                        f"Available columns: {list(adata.obs.columns)}")

    cell_types = adata.obs[cell_type_col].values

    # --- Step 2: Standardize gene symbols ---
    # Convert to uppercase (matching R's .transfer_symbol)
    gene_names = [g.upper() for g in gene_names]

    # Remove version numbers (e.g., "GENE.1" -> "GENE")
    gene_names = [g.split('.')[0] if '.' in g else g for g in gene_names]

    # --- Step 3: Handle duplicates (keep highest mean) ---
    if len(gene_names) != len(set(gene_names)):
        if verbose:
            print("  Removing duplicate genes (keeping highest mean)...")

        # Calculate mean per gene (axis=1 since matrix is now genes × cells)
        if sparse.issparse(counts):
            gene_means = np.asarray(counts.mean(axis=1)).ravel()
        else:
            gene_means = np.mean(counts, axis=1)

        # Keep best duplicate
        gene_to_best_idx = {}
        for idx, gene in enumerate(gene_names):
            if gene not in gene_to_best_idx or gene_means[idx] > gene_means[gene_to_best_idx[gene]]:
                gene_to_best_idx[gene] = idx

        keep_idx = sorted(gene_to_best_idx.values())
        counts = counts[keep_idx, :]
        gene_names = [gene_names[i] for i in keep_idx]

    n_genes = len(gene_names)
    n_cells = len(cell_names)

    if verbose:
        print(f"  Genes: {n_genes}, Cells: {n_cells}")
        print(f"  Cell types: {len(set(cell_types))}")

    # Check if data is already normalized (for later steps)
    if hasattr(counts, 'toarray'):
        sample_data = counts[:min(1000, counts.shape[0]), :min(100, counts.shape[1])].toarray()
    else:
        sample_data = counts[:min(1000, counts.shape[0]), :min(100, counts.shape[1])]
    
    max_val = np.max(sample_data)
    nonzero = sample_data[sample_data != 0]
    is_integer = len(nonzero) > 0 and np.allclose(nonzero, np.round(nonzero))
    is_already_normalized = max_val < 20 and not is_integer

    # --- Step 4: Aggregate or normalize ---
    if not is_single_cell_level:
        # R uses sort(unique(cellType_vec)) which is case-insensitive (locale-aware)
        # Python's sorted() is case-sensitive by default, so we use key=str.lower
        unique_cell_types = sorted(set(cell_types), key=str.lower)
        n_types = len(unique_cell_types)

        if is_already_normalized:
            # Data is already log-normalized: take MEAN per cell type
            if verbose:
                print("  Aggregating to pseudo-bulk by cell type (mean, data pre-normalized)...")
            
            pseudo_bulk = np.zeros((n_genes, n_types), dtype=np.float64)
            
            for j, ct in enumerate(unique_cell_types):
                mask = cell_types == ct
                cell_idx = np.where(mask)[0]
                
                if sparse.issparse(counts):
                    pseudo_bulk[:, j] = np.asarray(counts[:, cell_idx].mean(axis=1)).ravel()
                else:
                    pseudo_bulk[:, j] = counts[:, cell_idx].mean(axis=1)
            
            # Data is already normalized - use as-is
            expr = pseudo_bulk
            sample_names = unique_cell_types
            
            # Skip log transform (already done) - go directly to differential expression
            row_means = expr.mean(axis=1, keepdims=True)
            expr_diff = expr - row_means
            
        else:
            # Raw counts: sum by cell type, then normalize
            if verbose:
                print("  Aggregating to pseudo-bulk by cell type (sum)...")

            pseudo_bulk = np.zeros((n_genes, n_types), dtype=np.float64)

            for j, ct in enumerate(unique_cell_types):
                mask = cell_types == ct
                cell_idx = np.where(mask)[0]

                if sparse.issparse(counts):
                    pseudo_bulk[:, j] = np.asarray(counts[:, cell_idx].sum(axis=1)).ravel()
                else:
                    pseudo_bulk[:, j] = counts[:, cell_idx].sum(axis=1)

            # TPM normalize (counts per million)
            col_sums = pseudo_bulk.sum(axis=0)
            expr = pseudo_bulk / col_sums * 1e6

            sample_names = unique_cell_types
            
            # Log2 transform
            expr = np.log2(expr + 1)
            
            # Compute differential expression (vs row mean)
            row_means = expr.mean(axis=1, keepdims=True)
            expr_diff = expr - row_means

    else:
        # Single cell level: normalize per cell (counts per 100k)
        if verbose:
            print("  Normalizing per cell (CPM/10)...")

        if sparse_mode and sparse.issparse(counts):
            # --- Sparse end-to-end path ---
            # CPM and log2 are zero-preserving, so Y stays sparse
            col_sums = np.asarray(counts.sum(axis=0)).ravel()
            from scipy.sparse import diags as _sp_diags
            scaling = _sp_diags(1e5 / col_sums)
            expr_sparse = counts.astype(np.float64) @ scaling  # sparse CPM
            expr_sparse = expr_sparse.tocsc()
            expr_sparse.data = np.log2(expr_sparse.data + 1)  # sparse log2

            # Prepare signature (load, group, find common genes, z-score X)
            X_scaled, common_genes = _prepare_signature_for_sparse(
                gene_names=gene_names,
                sig_matrix=sig_matrix,
                is_group_sig=is_group_sig,
                is_group_cor=is_group_cor,
                sig_filter=sig_filter,
                sort_genes=sort_genes,
                verbose=verbose
            )

            # Subset sparse Y to common genes
            gene_to_idx = {g: i for i, g in enumerate(gene_names)}
            common_idx = [gene_to_idx[g] for g in common_genes]
            Y_sparse = expr_sparse[common_idx, :]

            if verbose:
                print(f"  Sparse Y: {Y_sparse.shape}, nnz={Y_sparse.nnz}, "
                      f"density={Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1]):.4f}")

            # Determine batch_size
            _batch_size = batch_size if batch_size is not None else 5000

            # Run ridge_batch directly with sparse_mode + row_center
            from .batch import ridge_batch
            result = ridge_batch(
                X=X_scaled.values,
                Y=Y_sparse,
                lambda_=lambda_,
                n_rand=n_rand,
                seed=seed,
                batch_size=_batch_size,
                backend=backend,
                use_gsl_rng=use_gsl_rng,
                rng_method=rng_method,
                use_cache=use_cache,
                output_path=output_path,
                output_compression=output_compression,
                feature_names=X_scaled.columns.tolist(),
                sample_names=cell_names,
                sparse_mode=True,
                row_center=True,
                verbose=verbose
            )

            # Format results
            if result is None:
                if verbose:
                    print(f"  Results streamed to {output_path}")
                return None

            feature_names = X_scaled.columns.tolist()
            beta_df = pd.DataFrame(result['beta'], index=feature_names, columns=cell_names)
            se_df = pd.DataFrame(result['se'], index=feature_names, columns=cell_names)
            zscore_df = pd.DataFrame(result['zscore'], index=feature_names, columns=cell_names)
            pvalue_df = pd.DataFrame(result['pvalue'], index=feature_names, columns=cell_names)

            if is_group_sig:
                if verbose:
                    print("  Expanding grouped signatures...")
                beta_df = expand_rows(beta_df)
                se_df = expand_rows(se_df)
                zscore_df = expand_rows(zscore_df)
                pvalue_df = expand_rows(pvalue_df)

                row_order = sorted(beta_df.index)
                beta_df = beta_df.loc[row_order]
                se_df = se_df.loc[row_order]
                zscore_df = zscore_df.loc[row_order]
                pvalue_df = pvalue_df.loc[row_order]

            if verbose:
                print(f"  Result shape: {beta_df.shape}")

            return {
                'beta': beta_df,
                'se': se_df,
                'zscore': zscore_df,
                'pvalue': pvalue_df
            }

        # --- Dense path (default) ---
        if sparse.issparse(counts):
            col_sums = np.asarray(counts.sum(axis=0)).ravel()
            expr = counts.toarray().astype(np.float64)
        else:
            col_sums = counts.sum(axis=0)
            expr = counts.astype(np.float64)

        expr = expr / col_sums * 1e5  # Counts per 100k (like R)

        sample_names = cell_names

        # Log2 transform
        expr = np.log2(expr + 1)

        # Compute differential expression (vs row mean)
        row_means = expr.mean(axis=1, keepdims=True)
        expr_diff = expr - row_means

    # Create DataFrame
    expr_df = pd.DataFrame(expr_diff, index=gene_names, columns=sample_names)

    if verbose:
        print(f"  Expression matrix: {expr_df.shape}")

    # --- Step 7: Run activity inference ---
    result = secact_activity_inference(
        input_profile=expr_df,
        is_differential=True,
        sig_matrix=sig_matrix,
        is_group_sig=is_group_sig,
        is_group_cor=is_group_cor,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        sig_filter=sig_filter,
        backend=backend,
        use_gsl_rng=use_gsl_rng,
        rng_method=rng_method,
        use_cache=use_cache,
        batch_size=batch_size,
        output_path=output_path,
        output_compression=output_compression,
        sort_genes=sort_genes,
        sparse_mode=sparse_mode,
        verbose=verbose
    )

    return result


# =============================================================================
# Spatial Transcriptomics Activity Inference (matching R's SecAct.activity.inference.ST)
# =============================================================================

def load_visium_10x(
    visium_path: str,
    min_genes: int = 0,
    verbose: bool = False
) -> dict[str, Any]:
    """
    Load 10X Visium spatial transcriptomics data.

    Matches R's create.SpaCET.object.10X behavior.

    Parameters
    ----------
    visium_path : str
        Path to Space Ranger output folder containing:
        - filtered_feature_bc_matrix/ (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
        - spatial/ (tissue_positions_list.csv, scalefactors_json.json)
    min_genes : int, default=0
        Minimum number of expressed genes per spot. Spots with fewer genes are removed.
    verbose : bool, default=False
        Print progress messages.

    Returns
    -------
    dict
        Dictionary with:
        - 'counts': scipy sparse matrix (genes × spots)
        - 'gene_names': list of gene names
        - 'spot_names': list of spot IDs (format: "row×col")
        - 'spot_coordinates': DataFrame with spatial coordinates
        - 'barcodes': list of original barcodes
    """
    from scipy.io import mmread
    import gzip
    import json

    visium_path = Path(visium_path)

    if not visium_path.exists():
        raise FileNotFoundError(f"Visium path does not exist: {visium_path}")

    # Load count matrix
    matrix_dir = visium_path / "filtered_feature_bc_matrix"

    if not matrix_dir.exists():
        raise FileNotFoundError(f"Matrix directory not found: {matrix_dir}")

    if verbose:
        print(f"Loading 10X Visium data from: {visium_path}")

    # Read matrix
    matrix_path = matrix_dir / "matrix.mtx.gz"
    with gzip.open(matrix_path, 'rb') as f:
        counts = mmread(f).tocsr()  # genes × spots

    if verbose:
        print(f"  Matrix shape: {counts.shape}")

    # Read gene names (features)
    features_path = matrix_dir / "features.tsv.gz"
    with gzip.open(features_path, 'rt') as f:
        features = [line.strip().split('\t') for line in f]

    # Use gene symbol (column 2 if available, else column 1)
    if len(features[0]) >= 2:
        gene_names = [f[1] for f in features]
    else:
        gene_names = [f[0] for f in features]

    # Read barcodes
    barcodes_path = matrix_dir / "barcodes.tsv.gz"
    with gzip.open(barcodes_path, 'rt') as f:
        barcodes = [line.strip() for line in f]

    if verbose:
        print(f"  Genes: {len(gene_names)}, Spots: {len(barcodes)}")

    # Load spatial information
    spatial_dir = visium_path / "spatial"

    # Try different position file names
    positions_file = None
    for fname in ["tissue_positions_list.csv", "tissue_positions.csv"]:
        pos_path = spatial_dir / fname
        if pos_path.exists():
            positions_file = pos_path
            break

    if positions_file is None:
        raise FileNotFoundError(f"Tissue positions file not found in: {spatial_dir}")

    # Read positions
    # Format: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    positions = pd.read_csv(positions_file, header=None)
    positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col',
                        'pxl_row_in_fullres', 'pxl_col_in_fullres']
    positions = positions.set_index('barcode')

    # Filter to in_tissue spots
    positions = positions[positions['in_tissue'] == 1]

    # Load scale factors
    scalefactors_path = spatial_dir / "scalefactors_json.json"
    if scalefactors_path.exists():
        with open(scalefactors_path, 'r') as f:
            scale_factors = json.load(f)
    else:
        scale_factors = {'tissue_lowres_scalef': 1.0}

    # Find overlapping spots
    common_barcodes = [b for b in barcodes if b in positions.index]
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
    common_idx = [barcode_to_idx[b] for b in common_barcodes]

    # Subset matrix
    counts = counts[:, common_idx]

    # Create spot IDs in R format: "row×col"
    spot_ids = []
    spot_coords = []
    for barcode in common_barcodes:
        row = positions.loc[barcode, 'array_row']
        col = positions.loc[barcode, 'array_col']
        spot_ids.append(f"{row}x{col}")
        spot_coords.append({
            'barcode': barcode,
            'array_row': row,
            'array_col': col,
            'pxl_row': positions.loc[barcode, 'pxl_row_in_fullres'],
            'pxl_col': positions.loc[barcode, 'pxl_col_in_fullres'],
        })

    spot_coordinates = pd.DataFrame(spot_coords, index=spot_ids)

    if verbose:
        print(f"  In-tissue spots: {len(spot_ids)}")

    # Apply quality control (min_genes filter)
    if min_genes > 0:
        genes_per_spot = np.asarray((counts > 0).sum(axis=0)).ravel()
        keep_spots = genes_per_spot >= min_genes

        n_removed = (~keep_spots).sum()
        if verbose:
            print(f"  Removing {n_removed} spots with < {min_genes} genes")

        counts = counts[:, keep_spots]
        spot_ids = [s for s, k in zip(spot_ids, keep_spots) if k]
        spot_coordinates = spot_coordinates.loc[spot_ids]
        common_barcodes = [b for b, k in zip(common_barcodes, keep_spots) if k]

    if verbose:
        print(f"  Final: {counts.shape[0]} genes × {counts.shape[1]} spots")

    return {
        'counts': counts,
        'gene_names': gene_names,
        'spot_names': spot_ids,
        'spot_coordinates': spot_coordinates,
        'barcodes': common_barcodes,
        'scale_factors': scale_factors,
    }


def secact_activity_inference_st(
    input_data,
    input_control = None,
    cell_type_col: Optional[str] = None,
    is_spot_level: bool = True,
    scale_factor: float = 1e5,
    sig_matrix: str = "secact",
    is_group_sig: bool = True,
    is_group_cor: float = 0.9,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    sig_filter: bool = False,
    min_genes: int = 0,
    backend: str = "auto",
    use_gsl_rng: bool = True,
    rng_method: Literal["srand", "gsl", "numpy", None] = None,
    use_cache: bool = False,
    batch_size: Optional[int] = None,
    output_path: Optional[str] = None,
    output_compression: Optional[str] = "gzip",
    sort_genes: bool = False,
    sparse_mode: bool = False,
    verbose: bool = False
) -> Optional[dict[str, Any]]:
    """
    Spot Activity Inference from Spatial Transcriptomics Data.

    Matches R's RidgeR::SecAct.activity.inference.ST behavior.

    Parameters
    ----------
    input_data : str, dict, or DataFrame
        One of:
        - Path to 10X Visium folder (str)
        - Result from load_visium_10x() (dict)
        - Count matrix as DataFrame (genes × spots)
        - AnnData object with spatial data
    input_control : optional
        Control expression data (same format as input_data).
        If None, uses mean of input_data as control.
    cell_type_col : str, optional
        Column name in AnnData.obs or metadata containing cell type annotations.
        If provided with is_spot_level=False, aggregates spots by cell type.
    is_spot_level : bool, default=True
        If True, compute activity for each spot individually.
        If False (and cell_type_col is provided), aggregate to pseudo-bulk by cell type.
    scale_factor : float, default=1e5
        Normalization scale factor (counts per scale_factor).
    sig_matrix : str, default="secact"
        Signature matrix: "secact", "cytosig", or path to custom file.
    is_group_sig : bool, default=True
        If True, group similar signatures by correlation.
    is_group_cor : float, default=0.9
        Correlation threshold for signature grouping.
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations for significance testing.
    seed : int, default=0
        Random seed for reproducibility.
    sig_filter : bool, default=False
        If True, filter signatures by available genes.
    min_genes : int, default=0
        Minimum genes per spot (only used if input_data is a path).
    backend : str, default="auto"
        Computation backend: "auto", "numpy", "cupy".
    use_gsl_rng : bool, default=True
        Use GSL-compatible RNG for exact R/RidgeR reproducibility.
        Set to False for faster inference when R matching is not needed.
    use_cache : bool, default=False
        Cache permutation tables to disk for reuse.
    output_path : str, optional
        Path to an HDF5 file for streaming results to disk. When set,
        results are written as each batch completes and the function
        returns ``None``. Requires ``batch_size``.
    output_compression : str, optional, default="gzip"
        Compression for the HDF5 output file ("gzip", "lzf", or None).
    sort_genes : bool, default=False
        If True, sort common genes alphabetically before running ridge regression.
        This ensures reproducible results across different platforms.
    verbose : bool, default=False
        Print progress messages.

    Returns
    -------
    dict or None
        If ``output_path`` is None, returns a dictionary with:

        - 'beta': DataFrame of coefficients (proteins × spots or proteins × cell_types)
        - 'se': DataFrame of standard errors
        - 'zscore': DataFrame of z-scores
        - 'pvalue': DataFrame of p-values

        If ``output_path`` is set, returns ``None``.

    Examples
    --------
    >>> # From 10X Visium folder (spot-level)
    >>> result = secact_activity_inference_st(
    ...     "path/to/visium/",
    ...     min_genes=1000,
    ...     verbose=True
    ... )
    >>>
    >>> # From AnnData with cell type annotations (cell-type resolution)
    >>> result = secact_activity_inference_st(
    ...     adata,
    ...     cell_type_col="cell_type",
    ...     is_spot_level=False,
    ...     verbose=True
    ... )
    >>>
    >>> # From count matrix
    >>> result = secact_activity_inference_st(counts_df)
    """
    from scipy import sparse

    if verbose:
        print("SecActPy Spatial Transcriptomics Activity Inference")
        print("=" * 50)

    # Track cell type annotations if available
    cell_types = None
    adata_obs = None

    # --- Step 1: Load/extract count matrix ---
    if isinstance(input_data, (str, Path)):
        input_path = str(input_data)
        # Check if it's an h5ad file or Visium folder
        if input_path.endswith('.h5ad'):
            import anndata as ad
            if verbose:
                print(f"  Loading: {input_path}")
            input_data = ad.read_h5ad(input_path)
            if verbose:
                print(f"  Loaded: {input_data.shape[0]} cells × {input_data.shape[1]} genes")
            
            # Apply min_genes filter to H5AD data
            if min_genes > 0:
                # Count expressed genes per cell/spot (genes > 0)
                if sparse.issparse(input_data.X):
                    genes_per_spot = np.asarray((input_data.X > 0).sum(axis=1)).ravel()
                else:
                    genes_per_spot = np.sum(input_data.X > 0, axis=1)
                
                keep_spots = genes_per_spot >= min_genes
                n_removed = (~keep_spots).sum()
                
                if verbose:
                    print(f"  Filtering: removing {n_removed} spots with < {min_genes} expressed genes")
                
                input_data = input_data[keep_spots, :].copy()
                
                if verbose:
                    print(f"  After filter: {input_data.shape[0]} spots")
            
            # Fall through to AnnData handling below
        else:
            # Path to Visium folder
            data = load_visium_10x(input_path, min_genes=min_genes, verbose=verbose)
            counts = data['counts']
            gene_names = data['gene_names']
            spot_names = data['spot_names']
            input_data = None  # Mark as handled

    if isinstance(input_data, dict) and 'counts' in input_data:
        # Result from load_visium_10x
        counts = input_data['counts']
        gene_names = input_data['gene_names']
        spot_names = input_data['spot_names']
    elif isinstance(input_data, pd.DataFrame):
        # DataFrame (genes × spots)
        counts = input_data.values
        gene_names = list(input_data.index)
        spot_names = list(input_data.columns)
    elif input_data is not None:
        # Try to handle AnnData
        try:
            # AnnData stores (cells/spots × genes)
            if hasattr(input_data, 'X') and hasattr(input_data, 'var_names'):
                if sparse.issparse(input_data.X):
                    counts = input_data.X.T.tocsr()  # Transpose to (genes × spots)
                else:
                    counts = input_data.X.T
                gene_names = list(input_data.var_names)
                spot_names = list(input_data.obs_names)
                adata_obs = input_data.obs
        except Exception:
            raise ValueError(
                "input_data must be a path to Visium folder, .h5ad file, dict from load_visium_10x(), "
                "DataFrame (genes × spots), or AnnData object."
            )

    if verbose:
        print(f"  Input: {len(gene_names)} genes × {len(spot_names)} spots")

    # --- Step 2: Remove zero-sum genes ---
    if sparse.issparse(counts):
        gene_sums = np.asarray(counts.sum(axis=1)).ravel()
    else:
        gene_sums = np.sum(counts, axis=1)

    nonzero_genes = gene_sums > 0
    counts = counts[nonzero_genes, :]
    gene_names = [g for g, keep in zip(gene_names, nonzero_genes) if keep]

    if verbose:
        print(f"  After removing zero genes: {len(gene_names)} genes")

    # --- Step 3: Standardize gene symbols ---
    # Convert to uppercase (matching R's .transfer_symbol)
    gene_names = [g.upper() for g in gene_names]

    # Remove version numbers (e.g., "GENE.1" -> "GENE")
    gene_names = [g.split('.')[0] if '.' in g else g for g in gene_names]

    # --- Step 4: Handle duplicates (keep highest sum) ---
    if len(gene_names) != len(set(gene_names)):
        if verbose:
            print("  Removing duplicate genes (keeping highest sum)...")

        if sparse.issparse(counts):
            gene_sums = np.asarray(counts.sum(axis=1)).ravel()
        else:
            gene_sums = np.sum(counts, axis=1)

        gene_to_best_idx = {}
        for idx, gene in enumerate(gene_names):
            if gene not in gene_to_best_idx or gene_sums[idx] > gene_sums[gene_to_best_idx[gene]]:
                gene_to_best_idx[gene] = idx

        keep_idx = sorted(gene_to_best_idx.values())
        counts = counts[keep_idx, :]
        gene_names = [gene_names[i] for i in keep_idx]

        if verbose:
            print(f"  After deduplication: {len(gene_names)} genes")

    n_genes = len(gene_names)

    # --- Step 5: Normalize (counts per scale_factor) ---
    # Sparse end-to-end path: spot-level, no control, sparse input
    if (sparse_mode and sparse.issparse(counts) and is_spot_level
            and input_control is None
            and (cell_type_col is None or is_spot_level)):
        # CPM and log2 are zero-preserving → Y stays sparse
        col_sums = np.asarray(counts.sum(axis=0)).ravel()
        from scipy.sparse import diags as _sp_diags
        scaling = _sp_diags(scale_factor / col_sums)
        expr_sparse = counts.astype(np.float64) @ scaling  # sparse CPM
        expr_sparse = expr_sparse.tocsc()
        expr_sparse.data = np.log2(expr_sparse.data + 1)  # sparse log2

        if verbose:
            print(f"  Normalized to counts per {scale_factor:.0e} (sparse)")

        # Prepare signature (load, group, find common genes, z-score X)
        X_scaled, common_genes = _prepare_signature_for_sparse(
            gene_names=gene_names,
            sig_matrix=sig_matrix,
            is_group_sig=is_group_sig,
            is_group_cor=is_group_cor,
            sig_filter=sig_filter,
            sort_genes=sort_genes,
            verbose=verbose
        )

        # Subset sparse Y to common genes
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        common_idx = [gene_to_idx[g] for g in common_genes]
        Y_sparse = expr_sparse[common_idx, :]

        if verbose:
            print(f"  Sparse Y: {Y_sparse.shape}, nnz={Y_sparse.nnz}, "
                  f"density={Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1]):.4f}")

        # Determine batch_size
        _batch_size = batch_size if batch_size is not None else 5000

        # Run ridge_batch directly with sparse_mode + row_center
        from .batch import ridge_batch
        result = ridge_batch(
            X=X_scaled.values,
            Y=Y_sparse,
            lambda_=lambda_,
            n_rand=n_rand,
            seed=seed,
            batch_size=_batch_size,
            backend=backend,
            use_gsl_rng=use_gsl_rng,
            rng_method=rng_method,
            use_cache=use_cache,
            output_path=output_path,
            output_compression=output_compression,
            feature_names=X_scaled.columns.tolist(),
            sample_names=spot_names,
            sparse_mode=True,
            row_center=True,
            verbose=verbose
        )

        # Format results
        if result is None:
            if verbose:
                print(f"  Results streamed to {output_path}")
            return None

        feature_names = X_scaled.columns.tolist()
        beta_df = pd.DataFrame(result['beta'], index=feature_names, columns=spot_names)
        se_df = pd.DataFrame(result['se'], index=feature_names, columns=spot_names)
        zscore_df = pd.DataFrame(result['zscore'], index=feature_names, columns=spot_names)
        pvalue_df = pd.DataFrame(result['pvalue'], index=feature_names, columns=spot_names)

        if is_group_sig:
            if verbose:
                print("  Expanding grouped signatures...")
            beta_df = expand_rows(beta_df)
            se_df = expand_rows(se_df)
            zscore_df = expand_rows(zscore_df)
            pvalue_df = expand_rows(pvalue_df)

            row_order = sorted(beta_df.index)
            beta_df = beta_df.loc[row_order]
            se_df = se_df.loc[row_order]
            zscore_df = zscore_df.loc[row_order]
            pvalue_df = pvalue_df.loc[row_order]

        if verbose:
            print(f"  Result shape: {beta_df.shape}")

        return {
            'beta': beta_df,
            'se': se_df,
            'zscore': zscore_df,
            'pvalue': pvalue_df
        }

    # --- Dense path (default) ---
    if sparse.issparse(counts):
        col_sums = np.asarray(counts.sum(axis=0)).ravel()
        expr = counts.toarray().astype(np.float64)
    else:
        col_sums = np.sum(counts, axis=0)
        expr = counts.astype(np.float64)

    # Normalize per spot
    expr = expr / col_sums * scale_factor

    if verbose:
        print(f"  Normalized to counts per {scale_factor:.0e}")

    # --- Step 6: Log2 transform ---
    expr = np.log2(expr + 1)

    # --- Step 6b: Cell type aggregation (if requested) ---
    sample_names = spot_names  # Default to spot names

    if cell_type_col is not None and not is_spot_level:
        # Validate cell type column
        if adata_obs is None:
            raise ValueError(
                f"cell_type_col='{cell_type_col}' requires AnnData input with obs metadata. "
                "For DataFrame or Visium folder input, use is_spot_level=True."
            )

        if cell_type_col not in adata_obs.columns:
            raise ValueError(
                f"Cell type column '{cell_type_col}' not found in adata.obs. "
                f"Available columns: {list(adata_obs.columns)}"
            )

        cell_types = adata_obs[cell_type_col].values

        if verbose:
            print(f"  Aggregating to pseudo-bulk by cell type ('{cell_type_col}')...")

        unique_cell_types = sorted(set(cell_types))
        n_types = len(unique_cell_types)

        if verbose:
            print(f"  Cell types: {n_types}")

        # Aggregate by cell type (mean of log-transformed values)
        pseudo_bulk = np.zeros((n_genes, n_types), dtype=np.float64)

        for j, ct in enumerate(unique_cell_types):
            mask = cell_types == ct
            spot_idx = np.where(mask)[0]
            pseudo_bulk[:, j] = expr[:, spot_idx].mean(axis=1)

        expr = pseudo_bulk
        sample_names = unique_cell_types

        if verbose:
            print(f"  Aggregated expression: {expr.shape[0]} genes × {expr.shape[1]} cell types")

    # --- Step 7: Compute differential expression ---
    if input_control is None:
        # Use row mean as control
        row_means = expr.mean(axis=1, keepdims=True)
        expr_diff = expr - row_means
    else:
        # Process control
        if isinstance(input_control, str):
            ctrl_data = load_visium_10x(input_control, verbose=False)
            ctrl_counts = ctrl_data['counts']
            ctrl_gene_names = ctrl_data['gene_names']
        elif isinstance(input_control, dict) and 'counts' in input_control:
            ctrl_counts = input_control['counts']
            ctrl_gene_names = input_control['gene_names']
        elif isinstance(input_control, pd.DataFrame):
            ctrl_counts = input_control.values
            ctrl_gene_names = list(input_control.index)
        else:
            raise ValueError("input_control format not recognized")

        # Standardize control gene names
        ctrl_gene_names = [g.upper().split('.')[0] for g in ctrl_gene_names]

        # Normalize and log2 transform control
        if sparse.issparse(ctrl_counts):
            ctrl_col_sums = np.asarray(ctrl_counts.sum(axis=0)).ravel()
            ctrl_expr = ctrl_counts.toarray().astype(np.float64)
        else:
            ctrl_col_sums = np.sum(ctrl_counts, axis=0)
            ctrl_expr = ctrl_counts.astype(np.float64)

        ctrl_expr = ctrl_expr / ctrl_col_sums * scale_factor
        ctrl_expr = np.log2(ctrl_expr + 1)

        # Find overlapping genes
        ctrl_gene_to_idx = {g: i for i, g in enumerate(ctrl_gene_names)}
        common_genes = [g for g in gene_names if g in ctrl_gene_to_idx]

        gene_idx = [gene_names.index(g) for g in common_genes]
        ctrl_idx = [ctrl_gene_to_idx[g] for g in common_genes]

        expr = expr[gene_idx, :]
        ctrl_mean = ctrl_expr[ctrl_idx, :].mean(axis=1, keepdims=True)
        expr_diff = expr - ctrl_mean
        gene_names = common_genes

    # Create DataFrame
    expr_df = pd.DataFrame(expr_diff, index=gene_names, columns=sample_names)

    if verbose:
        print(f"  Expression matrix: {expr_df.shape}")

    # --- Step 8: Run activity inference ---
    result = secact_activity_inference(
        input_profile=expr_df,
        is_differential=True,
        sig_matrix=sig_matrix,
        is_group_sig=is_group_sig,
        is_group_cor=is_group_cor,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        sig_filter=sig_filter,
        backend=backend,
        use_gsl_rng=use_gsl_rng,
        rng_method=rng_method,
        use_cache=use_cache,
        batch_size=batch_size,
        output_path=output_path,
        output_compression=output_compression,
        sort_genes=sort_genes,
        sparse_mode=sparse_mode,
        verbose=verbose
    )

    return result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SecActPy Inference Module - Testing")
    print("=" * 60)

    np.random.seed(42)

    # Create test data
    n_genes = 200
    n_features = 15
    n_samples = 10

    # Gene names (some shared, some not)
    all_genes = [f"GENE_{i}" for i in range(n_genes + 50)]
    expr_genes = all_genes[:n_genes]
    sig_genes = all_genes[25:n_genes + 25]  # Offset to test overlap

    # Create DataFrames
    expression = pd.DataFrame(
        np.random.randn(n_genes, n_samples),
        index=expr_genes,
        columns=[f"Sample_{i}" for i in range(n_samples)]
    )
    signature = pd.DataFrame(
        np.random.randn(len(sig_genes), n_features),
        index=sig_genes,
        columns=[f"Protein_{i}" for i in range(n_features)]
    )

    print("\nTest data:")
    print(f"  Expression: {expression.shape}")
    print(f"  Signature: {signature.shape}")

    # Test 1: Basic inference
    print("\n1. Testing basic inference...")
    result = secact_activity(
        expression, signature,
        n_rand=100, seed=0,
        verbose=True
    )

    print("\n   Results:")
    print(f"   - beta shape: {result['beta'].shape}")
    print(f"   - pvalue range: [{result['pvalue'].values.min():.4f}, {result['pvalue'].values.max():.4f}]")
    print(f"   - n_genes used: {result['n_genes']}")
    print(f"   - method: {result['method']}")

    # Verify DataFrame structure
    assert result['beta'].index.tolist() == result['features']
    assert result['beta'].columns.tolist() == result['samples']
    print("   ✓ DataFrame structure correct")

    # Test 2: T-test mode
    print("\n2. Testing t-test mode (n_rand=0)...")
    result_ttest = secact_activity(
        expression, signature,
        n_rand=0, seed=0
    )
    print(f"   - df: {result_ttest.get('df', 'N/A')}")
    print(f"   - time: {result_ttest['time']:.3f}s (vs {result['time']:.3f}s for permutation)")
    print("   ✓ T-test mode works")

    # Test 3: Scaling options
    print("\n3. Testing scaling options...")
    for scale in ["zscore", "center", "none"]:
        r = secact_activity(expression, signature, n_rand=50, scale=scale)
        print(f"   - scale='{scale}': beta mean={r['beta'].values.mean():.6f}")
    print("   ✓ All scaling options work")

    # Test 4: Reproducibility
    print("\n4. Testing reproducibility...")
    r1 = secact_activity(expression, signature, n_rand=100, seed=0)
    r2 = secact_activity(expression, signature, n_rand=100, seed=0)

    if np.allclose(r1['beta'].values, r2['beta'].values) and \
       np.allclose(r1['pvalue'].values, r2['pvalue'].values):
        print("   ✓ Results reproducible with same seed")
    else:
        print("   ✗ Results not reproducible!")

    # Test 5: Differential expression helper
    print("\n5. Testing differential expression helper...")
    control = pd.DataFrame(
        np.random.randn(n_genes, 5),
        index=expr_genes,
        columns=[f"Ctrl_{i}" for i in range(5)]
    )

    diff = compute_differential(expression, control, aggregate=True)
    print(f"   - Differential shape: {diff.shape}")

    diff_paired = compute_differential(
        expression.iloc[:, :5].rename(columns=lambda x: x.replace("Sample", "Pair")),
        control.rename(columns=lambda x: x.replace("Ctrl", "Pair")),
        paired=True, aggregate=False
    )
    print(f"   - Paired differential shape: {diff_paired.shape}")
    print("   ✓ Differential expression helper works")

    # Test 6: Edge case - few genes
    print("\n6. Testing edge case (few common genes)...")
    small_sig = signature.iloc[:15]  # Only 15 genes
    try:
        r_small = secact_activity(expression, small_sig, n_rand=50, min_genes=10)
        print(f"   - Used {r_small['n_genes']} genes")
        print("   ✓ Works with few genes")
    except ValueError as e:
        print(f"   - Correctly raised error: {e}")

    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
