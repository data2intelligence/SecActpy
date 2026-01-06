"""
I/O utilities for SecActPy results.

This module provides functions for:
- Loading results from HDF5/h5ad files (created by ridge_batch streaming)
- Converting results to AnnData format for integration with scanpy
- Saving results to various formats

File Formats:
-------------
- HDF5 (.h5, .h5ad): Native format for streaming output
- AnnData: Scanpy-compatible format for downstream analysis
- CSV/Parquet: For export to other tools

Usage:
------
    >>> from secactpy.io import load_results, results_to_anndata
    >>>
    >>> # Load streaming results
    >>> results = load_results("results.h5ad")
    >>>
    >>> # Convert to AnnData for scanpy
    >>> adata = results_to_anndata(results)
    >>>
    >>> # Or load directly as AnnData
    >>> adata = load_as_anndata("results.h5ad")
"""

import numpy as np
import pandas as pd
from typing import Optional, Any, Union, List
from pathlib import Path
import warnings

__all__ = [
    'load_results',
    'save_results',
    'save_results_to_h5ad',
    'results_to_anndata',
    'load_as_anndata',
    'results_to_dataframes',
    'save_st_results_to_h5ad',
    'add_activity_to_anndata',
    'H5PY_AVAILABLE',
    'ANNDATA_AVAILABLE',
]


# =============================================================================
# Optional Dependencies
# =============================================================================

H5PY_AVAILABLE = False
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    pass

ANNDATA_AVAILABLE = False
try:
    import anndata
    ANNDATA_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Loading Functions
# =============================================================================

def load_results(
    path: Union[str, Path],
    load_arrays: bool = True,
    mmap_mode: Optional[str] = None
) -> dict[str, Any]:
    """
    Load results from HDF5 file created by ridge_batch.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file (.h5 or .h5ad).
    load_arrays : bool, default=True
        If True, load full arrays into memory.
        If False, return h5py dataset references (for lazy loading).
    mmap_mode : str, optional
        Memory-map mode for numpy arrays ('r', 'r+', 'c').
        Only used if load_arrays=True.

    Returns
    -------
    dict
        Results dictionary containing:
        - beta, se, zscore, pvalue: ndarrays or h5py datasets
        - feature_names: list of feature names (if stored)
        - sample_names: list of sample names (if stored)
        - attrs: dict of file attributes

    Examples
    --------
    >>> results = load_results("output.h5ad")
    >>> print(results['beta'].shape)
    (50, 100000)

    >>> # Lazy loading for very large files
    >>> results = load_results("large_output.h5ad", load_arrays=False)
    >>> # Access slices without loading full array
    >>> batch = results['beta'][:, :1000]
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py required for loading HDF5 files. Install with: pip install h5py")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    result = {}

    # Open file
    f = h5py.File(path, 'r')

    # Load or reference arrays
    array_names = ['beta', 'se', 'zscore', 'pvalue']
    for name in array_names:
        if name in f:
            if load_arrays:
                result[name] = f[name][:]
            else:
                result[name] = f[name]  # Keep as h5py dataset

    # Load metadata
    result['attrs'] = dict(f.attrs)

    # Decode feature/sample names
    if 'feature_names' in f.attrs:
        names = f.attrs['feature_names']
        if isinstance(names[0], bytes):
            result['feature_names'] = [n.decode('utf-8') for n in names]
        else:
            result['feature_names'] = list(names)

    if 'sample_names' in f.attrs:
        names = f.attrs['sample_names']
        if isinstance(names[0], bytes):
            result['sample_names'] = [n.decode('utf-8') for n in names]
        else:
            result['sample_names'] = list(names)

    # Keep file handle for lazy loading
    if not load_arrays:
        result['_file'] = f
    else:
        f.close()

    return result


def results_to_dataframes(
    results: dict[str, Any],
    feature_names: Optional[List[str]] = None,
    sample_names: Optional[List[str]] = None
) -> dict[str, pd.DataFrame]:
    """
    Convert results arrays to labeled pandas DataFrames.

    Parameters
    ----------
    results : dict
        Results dictionary with 'beta', 'se', 'zscore', 'pvalue' arrays.
    feature_names : list, optional
        Feature names for row index. If None, uses results['feature_names']
        or generates default names.
    sample_names : list, optional
        Sample names for column index. If None, uses results['sample_names']
        or generates default names.

    Returns
    -------
    dict
        Dictionary with 'beta', 'se', 'zscore', 'pvalue' as DataFrames.
    """
    # Get array shape
    beta = results.get('beta')
    if beta is None:
        raise ValueError("Results must contain 'beta' array")

    n_features, n_samples = beta.shape

    # Get or generate names
    if feature_names is None:
        feature_names = results.get('feature_names',
                                     [f"Feature_{i}" for i in range(n_features)])
    if sample_names is None:
        sample_names = results.get('sample_names',
                                    [f"Sample_{i}" for i in range(n_samples)])

    # Convert to DataFrames
    dfs = {}
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        if name in results:
            arr = results[name]
            # Handle h5py datasets
            if hasattr(arr, 'shape') and not isinstance(arr, np.ndarray):
                arr = arr[:]
            dfs[name] = pd.DataFrame(arr, index=feature_names, columns=sample_names)

    return dfs


# =============================================================================
# AnnData Conversion
# =============================================================================

def results_to_anndata(
    results: dict[str, Any],
    feature_names: Optional[List[str]] = None,
    sample_names: Optional[List[str]] = None,
    primary_layer: str = 'zscore'
) -> 'anndata.AnnData':
    """
    Convert results to AnnData format for scanpy integration.

    The AnnData object is structured as:
    - obs (rows): samples
    - var (columns): features/proteins
    - X: primary result layer (default: zscore)
    - layers: all result arrays (beta, se, zscore, pvalue)

    Parameters
    ----------
    results : dict
        Results dictionary from ridge_batch or load_results.
    feature_names : list, optional
        Feature names. If None, uses stored names or generates defaults.
    sample_names : list, optional
        Sample names. If None, uses stored names or generates defaults.
    primary_layer : str, default='zscore'
        Which result array to use as the main X matrix.

    Returns
    -------
    AnnData
        AnnData object with samples as obs and features as var.
        - adata.X: primary layer (zscore by default)
        - adata.layers['beta']: coefficients
        - adata.layers['se']: standard errors
        - adata.layers['zscore']: z-scores
        - adata.layers['pvalue']: p-values

    Examples
    --------
    >>> results = load_results("output.h5ad")
    >>> adata = results_to_anndata(results)
    >>>
    >>> # Use with scanpy
    >>> import scanpy as sc
    >>> sc.pl.heatmap(adata, var_names=adata.var_names[:10])

    Notes
    -----
    The output is transposed from the internal representation:
    - Internal: (n_features, n_samples)
    - AnnData: (n_samples, n_features) - samples as obs
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError(
            "anndata required for AnnData conversion. "
            "Install with: pip install anndata"
        )

    # Get array
    beta = results.get('beta')
    if beta is None:
        raise ValueError("Results must contain 'beta' array")

    # Handle h5py datasets
    if hasattr(beta, 'shape') and not isinstance(beta, np.ndarray):
        beta = beta[:]

    n_features, n_samples = beta.shape

    # Get or generate names
    if feature_names is None:
        feature_names = results.get('feature_names',
                                     [f"Feature_{i}" for i in range(n_features)])
    if sample_names is None:
        sample_names = results.get('sample_names',
                                    [f"Sample_{i}" for i in range(n_samples)])

    # Get primary layer
    primary = results.get(primary_layer)
    if primary is None:
        primary = beta
        warnings.warn(f"Primary layer '{primary_layer}' not found, using 'beta'")
    if hasattr(primary, 'shape') and not isinstance(primary, np.ndarray):
        primary = primary[:]

    # Create AnnData (transposed: samples as obs)
    adata = anndata.AnnData(
        X=primary.T,  # (n_samples, n_features)
        obs=pd.DataFrame(index=sample_names),
        var=pd.DataFrame(index=feature_names)
    )

    # Add all layers
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        if name in results:
            arr = results[name]
            if hasattr(arr, 'shape') and not isinstance(arr, np.ndarray):
                arr = arr[:]
            adata.layers[name] = arr.T  # Transpose to (n_samples, n_features)

    # Add any additional attributes
    if 'attrs' in results:
        for key, value in results['attrs'].items():
            if key not in ['feature_names', 'sample_names']:
                adata.uns[key] = value

    return adata


def save_st_results_to_h5ad(
    counts,
    activity_results: dict[str, Any],
    output_path: Union[str, Path],
    gene_names: Optional[List[str]] = None,
    cell_names: Optional[List[str]] = None,
    spatial_coords: Optional[pd.DataFrame] = None,
    metadata: Optional[pd.DataFrame] = None,
    platform: str = "unknown"
) -> 'anndata.AnnData':
    """
    Save spatial transcriptomics data with activity results to h5ad format.

    Creates a combined AnnData object containing:
    - Raw counts in adata.X (or adata.raw.X)
    - Activity results in adata.obsm (SecAct_beta, SecAct_zscore, etc.)
    - Spatial coordinates in adata.obsm['spatial']
    - Metadata in adata.obs

    Parameters
    ----------
    counts : array-like or sparse matrix
        Count matrix (genes × cells). Will be transposed for AnnData format.
    activity_results : dict
        Results from secact_activity_inference_st containing:
        - 'beta': DataFrame (proteins × cells)
        - 'se': DataFrame (proteins × cells)
        - 'zscore': DataFrame (proteins × cells)
        - 'pvalue': DataFrame (proteins × cells)
    output_path : str or Path
        Output path for .h5ad file.
    gene_names : list, optional
        Gene names. If None, uses integers.
    cell_names : list, optional
        Cell/spot names. If None, uses integers.
    spatial_coords : DataFrame, optional
        Spatial coordinates with columns for x, y positions.
        Index should match cell_names.
    metadata : DataFrame, optional
        Additional cell/spot metadata.
        Index should match cell_names.
    platform : str, default="unknown"
        Platform name (e.g., "Visium", "CosMx").

    Returns
    -------
    AnnData
        The created AnnData object.

    Examples
    --------
    >>> from secactpy import secact_activity_inference_st, load_visium_10x
    >>> from secactpy.io import save_st_results_to_h5ad
    >>>
    >>> # Load and run inference
    >>> data = load_visium_10x("visium_folder/", min_genes=1000)
    >>> results = secact_activity_inference_st(data, verbose=True)
    >>>
    >>> # Save combined data
    >>> adata = save_st_results_to_h5ad(
    ...     counts=data['counts'],
    ...     activity_results=results,
    ...     output_path="visium_with_activity.h5ad",
    ...     gene_names=data['gene_names'],
    ...     cell_names=data['spot_names'],
    ...     spatial_coords=data['spot_coordinates'],
    ...     platform="Visium"
    ... )

    Notes
    -----
    The activity results are stored in adata.obsm as:
    - adata.obsm['SecAct_beta']: (cells × proteins)
    - adata.obsm['SecAct_se']: (cells × proteins)
    - adata.obsm['SecAct_zscore']: (cells × proteins)
    - adata.obsm['SecAct_pvalue']: (cells × proteins)

    Protein names are stored in adata.uns['SecAct_protein_names'].
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError(
            "anndata required. Install with: pip install anndata"
        )

    from scipy import sparse

    output_path = Path(output_path)

    # Handle sparse matrices
    if sparse.issparse(counts):
        # Transpose to (cells × genes) for AnnData
        X = counts.T.tocsr()
    else:
        X = np.asarray(counts).T

    n_cells, n_genes = X.shape

    # Default names
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
    if cell_names is None:
        cell_names = [f"Cell_{i}" for i in range(n_cells)]

    # Create obs DataFrame
    if metadata is not None:
        obs = metadata.copy()
        obs.index = cell_names
    else:
        obs = pd.DataFrame(index=cell_names)

    # Add platform info
    obs['platform'] = platform

    # Create var DataFrame
    var = pd.DataFrame(index=gene_names)

    # Create AnnData
    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=var
    )

    # Add spatial coordinates
    if spatial_coords is not None:
        # Extract x, y coordinates
        if 'pixel_row' in spatial_coords.columns and 'pixel_col' in spatial_coords.columns:
            spatial_array = spatial_coords[['pixel_col', 'pixel_row']].values
        elif 'x' in spatial_coords.columns and 'y' in spatial_coords.columns:
            spatial_array = spatial_coords[['x', 'y']].values
        elif spatial_coords.shape[1] >= 2:
            spatial_array = spatial_coords.iloc[:, :2].values
        else:
            spatial_array = None

        if spatial_array is not None:
            # Ensure correct order
            if len(spatial_coords) == n_cells:
                adata.obsm['spatial'] = spatial_array
            else:
                # Reindex to match cell_names
                try:
                    aligned_coords = spatial_coords.loc[cell_names]
                    if 'pixel_row' in aligned_coords.columns:
                        adata.obsm['spatial'] = aligned_coords[['pixel_col', 'pixel_row']].values
                    else:
                        adata.obsm['spatial'] = aligned_coords.iloc[:, :2].values
                except KeyError:
                    warnings.warn("Could not align spatial coordinates with cell names")

    # Add activity results to obsm
    # Activity results are (proteins × cells), need to transpose to (cells × proteins)
    protein_names = None

    for result_name in ['beta', 'se', 'zscore', 'pvalue']:
        if result_name in activity_results:
            result_df = activity_results[result_name]

            if isinstance(result_df, pd.DataFrame):
                # Get protein names from first result
                if protein_names is None:
                    protein_names = list(result_df.index)

                # Align columns to cell_names
                if set(result_df.columns) == set(cell_names):
                    result_aligned = result_df[cell_names].T  # (cells × proteins)
                else:
                    # Try to find matching columns
                    common_cells = [c for c in cell_names if c in result_df.columns]
                    if len(common_cells) == len(cell_names):
                        result_aligned = result_df[common_cells].T
                    else:
                        warnings.warn(
                            f"Activity result columns don't match cell names for {result_name}. "
                            f"Expected {len(cell_names)}, got {len(result_df.columns)} columns."
                        )
                        result_aligned = result_df.T

                adata.obsm[f'SecAct_{result_name}'] = result_aligned.values
            else:
                # Assume it's an array (proteins × cells)
                adata.obsm[f'SecAct_{result_name}'] = np.asarray(result_df).T

    # Store protein names
    if protein_names is not None:
        adata.uns['SecAct_protein_names'] = protein_names

    # Store platform info
    adata.uns['platform'] = platform
    adata.uns['secactpy_version'] = "0.1.0"

    # Save to h5ad
    adata.write_h5ad(output_path)

    print(f"Saved to: {output_path}")
    print(f"  Shape: {adata.shape} (cells × genes)")
    print(f"  Activity results: {[k for k in adata.obsm.keys() if k.startswith('SecAct_')]}")
    if protein_names:
        print(f"  Proteins: {len(protein_names)}")

    return adata


def add_activity_to_anndata(
    adata: 'anndata.AnnData',
    activity_results: dict[str, Any],
    key_prefix: str = "SecAct",
    layer_name: Optional[str] = "SecAct_zscore",
    copy: bool = False
) -> 'anndata.AnnData':
    """
    Add activity inference results to an existing AnnData object.

    For single-cell level analysis, activity results (proteins × cells) are added to:
    - adata.obsm: Activity matrices as (cells × proteins)
    - adata.uns: Protein names and metadata
    - adata.layers (optional): Z-scores as a layer for visualization

    Parameters
    ----------
    adata : AnnData
        AnnData object to add activity results to.
    activity_results : dict
        Results from secact_activity_inference_scrnaseq containing:
        - 'beta': DataFrame (proteins × cells)
        - 'se': DataFrame (proteins × cells)
        - 'zscore': DataFrame (proteins × cells)
        - 'pvalue': DataFrame (proteins × cells)
    key_prefix : str, default="SecAct"
        Prefix for obsm keys (e.g., "SecAct_zscore", "SecAct_beta").
    layer_name : str or None, default="SecAct_zscore"
        If provided, also add z-scores as a layer for easy plotting.
        Set to None to skip layer creation.
    copy : bool, default=False
        If True, return a copy of the AnnData object.
        If False, modify in place.

    Returns
    -------
    AnnData
        AnnData object with activity results added.

    Examples
    --------
    >>> import anndata as ad
    >>> from secactpy import secact_activity_inference_scrnaseq
    >>> from secactpy.io import add_activity_to_anndata
    >>>
    >>> # Load data and run single-cell inference
    >>> adata = ad.read_h5ad("scrnaseq_data.h5ad")
    >>> results = secact_activity_inference_scrnaseq(
    ...     adata,
    ...     cell_type_col="cell_type",
    ...     is_single_cell_level=True
    ... )
    >>>
    >>> # Add results to AnnData
    >>> adata = add_activity_to_anndata(adata, results)
    >>>
    >>> # Now you can use scanpy for visualization
    >>> import scanpy as sc
    >>> # Z-scores are in adata.obsm['SecAct_zscore']
    >>> # Protein names are in adata.uns['SecAct_protein_names']
    >>>
    >>> # Plot activity on UMAP
    >>> protein_idx = adata.uns['SecAct_protein_names'].index('IL6')
    >>> adata.obs['IL6_activity'] = adata.obsm['SecAct_zscore'][:, protein_idx]
    >>> sc.pl.umap(adata, color='IL6_activity')
    >>>
    >>> # Save with results
    >>> adata.write_h5ad("scrnaseq_with_activity.h5ad")

    Notes
    -----
    Activity matrices in obsm are stored as (cells × proteins), transposed from
    the inference output format (proteins × cells).

    The protein names can be accessed via:
        protein_names = adata.uns['SecAct_protein_names']

    To get activity for a specific protein:
        protein_idx = protein_names.index('IL6')
        il6_activity = adata.obsm['SecAct_zscore'][:, protein_idx]
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError(
            "anndata required. Install with: pip install anndata"
        )

    if copy:
        adata = adata.copy()

    # Get cell names from AnnData
    cell_names = list(adata.obs_names)
    n_cells = len(cell_names)

    # Get protein names from first result
    protein_names = None

    for result_name in ['beta', 'se', 'zscore', 'pvalue']:
        if result_name not in activity_results:
            continue

        result_df = activity_results[result_name]

        if isinstance(result_df, pd.DataFrame):
            # Get protein names
            if protein_names is None:
                protein_names = list(result_df.index)

            # Check if columns match cells
            result_cols = list(result_df.columns)

            if len(result_cols) != n_cells:
                raise ValueError(
                    f"Activity results have {len(result_cols)} columns but AnnData has {n_cells} cells. "
                    "For pseudo-bulk results (per cell-type), use save_results() instead."
                )

            # Align columns to cell order if needed
            if result_cols != cell_names:
                # Check if it's just a different order
                if set(result_cols) == set(cell_names):
                    result_df = result_df[cell_names]
                else:
                    warnings.warn(
                        "Cell names in activity results don't exactly match AnnData. "
                        "Using activity column order."
                    )

            # Transpose to (cells × proteins) and add to obsm
            adata.obsm[f'{key_prefix}_{result_name}'] = result_df.T.values

        else:
            # Assume numpy array (proteins × cells)
            arr = np.asarray(result_df)
            if arr.shape[1] != n_cells:
                raise ValueError(
                    f"Activity array has {arr.shape[1]} columns but AnnData has {n_cells} cells."
                )
            adata.obsm[f'{key_prefix}_{result_name}'] = arr.T

    # Store protein names
    if protein_names is not None:
        adata.uns[f'{key_prefix}_protein_names'] = protein_names
        adata.uns[f'{key_prefix}_n_proteins'] = len(protein_names)

    # Store metadata
    adata.uns[f'{key_prefix}_inference_type'] = 'single_cell'
    adata.uns['secactpy_version'] = "0.1.0"

    # Optionally add z-scores as a layer (for easy visualization)
    # Note: layers must have shape (n_obs, n_vars), but our activity is (n_obs, n_proteins)
    # So we can't add it as a standard layer. Instead, we document how to use obsm.

    print("Added activity results to AnnData:")
    print(f"  Cells: {n_cells}")
    print(f"  Proteins: {len(protein_names) if protein_names else 'unknown'}")
    print(f"  Added to obsm: {[k for k in adata.obsm.keys() if k.startswith(key_prefix)]}")
    print(f"  Protein names in: adata.uns['{key_prefix}_protein_names']")

    return adata


def load_as_anndata(
    path: Union[str, Path],
    primary_layer: str = 'zscore'
) -> 'anndata.AnnData':
    """
    Load HDF5 results directly as AnnData.

    Convenience function combining load_results and results_to_anndata.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.
    primary_layer : str, default='zscore'
        Which layer to use as X matrix.

    Returns
    -------
    AnnData
        AnnData object with results.
    """
    results = load_results(path, load_arrays=True)
    return results_to_anndata(results, primary_layer=primary_layer)


# =============================================================================
# Saving Functions
# =============================================================================

def save_results(
    results: dict[str, Any],
    path: Union[str, Path],
    format: str = 'auto',
    compression: Optional[str] = 'gzip',
    feature_names: Optional[List[str]] = None,
    sample_names: Optional[List[str]] = None
) -> None:
    """
    Save results to file.

    Parameters
    ----------
    results : dict
        Results dictionary with 'beta', 'se', 'zscore', 'pvalue' arrays.
    path : str or Path
        Output path.
    format : str, default='auto'
        Output format: 'h5', 'h5ad', 'csv', 'parquet', or 'auto' (infer from extension).
    compression : str, optional
        Compression for HDF5 ('gzip', 'lzf', None).
    feature_names : list, optional
        Feature names to store.
    sample_names : list, optional
        Sample names to store.
    """
    path = Path(path)

    # Auto-detect format
    if format == 'auto':
        suffix = path.suffix.lower()
        if suffix in ['.h5', '.hdf5', '.h5ad']:
            format = 'h5'
        elif suffix == '.csv':
            format = 'csv'
        elif suffix == '.parquet':
            format = 'parquet'
        else:
            format = 'h5'  # Default

    if format in ['h5', 'h5ad', 'hdf5']:
        _save_hdf5(results, path, compression, feature_names, sample_names)
    elif format == 'csv':
        _save_csv(results, path, feature_names, sample_names)
    elif format == 'parquet':
        _save_parquet(results, path, feature_names, sample_names)
    else:
        raise ValueError(f"Unknown format: {format}")


def _save_hdf5(
    results: dict[str, Any],
    path: Path,
    compression: Optional[str],
    feature_names: Optional[List[str]],
    sample_names: Optional[List[str]]
) -> None:
    """Save results to HDF5."""
    if not H5PY_AVAILABLE:
        raise ImportError("h5py required. Install with: pip install h5py")

    with h5py.File(path, 'w') as f:
        # Save arrays
        for name in ['beta', 'se', 'zscore', 'pvalue']:
            if name in results:
                arr = results[name]
                if hasattr(arr, 'shape') and not isinstance(arr, np.ndarray):
                    arr = arr[:]
                f.create_dataset(name, data=arr, compression=compression)

        # Save names
        if feature_names is not None:
            f.attrs['feature_names'] = np.array(feature_names, dtype='S')
        elif 'feature_names' in results:
            f.attrs['feature_names'] = np.array(results['feature_names'], dtype='S')

        if sample_names is not None:
            f.attrs['sample_names'] = np.array(sample_names, dtype='S')
        elif 'sample_names' in results:
            f.attrs['sample_names'] = np.array(results['sample_names'], dtype='S')


def _save_csv(
    results: dict[str, Any],
    path: Path,
    feature_names: Optional[List[str]],
    sample_names: Optional[List[str]]
) -> None:
    """Save results to CSV files (one per array)."""
    dfs = results_to_dataframes(results, feature_names, sample_names)

    # Create directory for multiple files
    base = path.stem
    parent = path.parent

    for name, df in dfs.items():
        out_path = parent / f"{base}_{name}.csv"
        df.to_csv(out_path)


def _save_parquet(
    results: dict[str, Any],
    path: Path,
    feature_names: Optional[List[str]],
    sample_names: Optional[List[str]]
) -> None:
    """Save results to Parquet files."""
    import importlib.util
    if importlib.util.find_spec("pyarrow") is None:
        raise ImportError("pyarrow required for parquet. Install with: pip install pyarrow")

    dfs = results_to_dataframes(results, feature_names, sample_names)

    base = path.stem
    parent = path.parent

    for name, df in dfs.items():
        out_path = parent / f"{base}_{name}.parquet"
        df.to_parquet(out_path)


def save_results_to_h5ad(
    results: dict[str, Any],
    path: Union[str, Path],
    feature_names: Optional[List[str]] = None,
    sample_names: Optional[List[str]] = None,
    compression: str = 'gzip',
    verbose: bool = False
) -> None:
    """
    Save results to AnnData-compatible H5AD format.

    Creates an H5AD file with:
    - X: beta coefficients (samples × features)
    - obsm/se: standard errors
    - obsm/zscore: z-scores
    - obsm/pvalue: p-values
    - obs/_index: sample names
    - var/_index: feature names

    Parameters
    ----------
    results : dict
        Results dictionary with 'beta', 'se', 'zscore', 'pvalue' arrays.
        Arrays should be (features × samples) shaped.
    path : str or Path
        Output path for H5AD file.
    feature_names : list, optional
        Feature/protein names. Uses results['feature_names'] if available.
    sample_names : list, optional
        Sample/cell names. Uses results['sample_names'] if available.
    compression : str, default='gzip'
        Compression algorithm.
    verbose : bool, default=False
        Print progress messages.

    Examples
    --------
    >>> from secactpy import secact_activity_inference
    >>> from secactpy.io import save_results_to_h5ad
    >>>
    >>> results = secact_activity_inference(expr, verbose=True)
    >>> save_results_to_h5ad(results, "output.h5ad", verbose=True)

    Notes
    -----
    The H5AD file can be loaded in Python with:
        >>> import anndata
        >>> adata = anndata.read_h5ad("output.h5ad")
        >>> beta = adata.X  # (samples × features)
        >>> zscore = adata.obsm['zscore']
    """
    path = Path(path)

    # Extract arrays
    beta = results.get('beta')
    if beta is None:
        raise ValueError("Results must contain 'beta' array")

    # Handle DataFrame input
    if hasattr(beta, 'values'):
        if feature_names is None:
            feature_names = list(beta.index)
        if sample_names is None:
            sample_names = list(beta.columns)
        beta = beta.values

    n_features, n_samples = beta.shape

    # Get names
    if feature_names is None:
        feature_names = results.get('feature_names', [f"Feature_{i}" for i in range(n_features)])
    if sample_names is None:
        sample_names = results.get('sample_names', [f"Sample_{i}" for i in range(n_samples)])

    if verbose:
        print(f"Saving results to H5AD: {path}")
        print(f"  Features: {n_features}")
        print(f"  Samples:  {n_samples}")

    # Remove existing file
    if path.exists():
        path.unlink()

    # Prefer anndata when available (creates fully valid h5ad files)
    if ANNDATA_AVAILABLE:
        _save_with_anndata(
            results, path, feature_names, sample_names, 
            n_features, n_samples, compression, verbose
        )
    elif H5PY_AVAILABLE:
        _save_with_h5py(
            results, path, feature_names, sample_names,
            n_features, n_samples, compression, verbose
        )
    else:
        raise ImportError("Either anndata or h5py required. Install with: pip install anndata h5py")


def _save_with_anndata(
    results: dict[str, Any],
    path: Path,
    feature_names: List[str],
    sample_names: List[str],
    n_features: int,
    n_samples: int,
    compression: str,
    verbose: bool
) -> None:
    """Save using anndata - creates fully valid h5ad files."""
    import anndata
    
    # Get arrays and transpose to (samples × features)
    beta = results['beta']
    if hasattr(beta, 'values'):
        beta = beta.values
    X = beta.T.astype(np.float64)
    
    # Create AnnData object
    adata = anndata.AnnData(X=X)
    
    # Set names
    adata.obs_names = pd.Index(sample_names)
    adata.var_names = pd.Index(feature_names)
    
    # Add obsm matrices
    for name in ['se', 'zscore', 'pvalue']:
        if name in results:
            arr = results[name]
            if hasattr(arr, 'values'):
                arr = arr.values
            arr_t = arr.T.astype(np.float64)
            adata.obsm[name] = arr_t
    
    # Add uns metadata
    adata.uns['source'] = 'SecActPy'
    adata.uns['feature_names'] = list(feature_names)
    
    # Write file
    adata.write_h5ad(path, compression=compression)
    
    if verbose:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        print(f"\nPython usage:")
        print(f"  import anndata")
        print(f"  adata = anndata.read_h5ad('{path.name}')")
        print(f"  beta = adata.X  # ({n_samples} × {n_features})")
        print(f"  zscore = adata.obsm['zscore']")


def _save_with_h5py(
    results: dict[str, Any],
    path: Path,
    feature_names: List[str],
    sample_names: List[str],
    n_features: int,
    n_samples: int,
    compression: str,
    verbose: bool
) -> None:
    """Save using h5py - fallback when anndata not available."""
    import h5py
    
    beta = results['beta']
    if hasattr(beta, 'values'):
        beta = beta.values

    with h5py.File(path, 'w') as f:
        # Create groups
        f.create_group('obs')
        f.create_group('var')
        f.create_group('obsm')
        f.create_group('uns')

        # Write X (beta, transposed: samples × features)
        beta_t = beta.T if isinstance(beta, np.ndarray) else np.array(beta).T
        f.create_dataset('X', data=beta_t, compression=compression)

        # Write obsm (se, zscore, pvalue)
        for name in ['se', 'zscore', 'pvalue']:
            if name in results:
                arr = results[name]
                if hasattr(arr, 'values'):
                    arr = arr.values
                arr_t = arr.T if isinstance(arr, np.ndarray) else np.array(arr).T
                f.create_dataset(f'obsm/{name}', data=arr_t, compression=compression)

        # Write indices
        f.create_dataset('obs/_index', data=np.array(sample_names, dtype='S'))
        f.create_dataset('var/_index', data=np.array(feature_names, dtype='S'))

        # Write uns metadata
        f.create_dataset('uns/feature_names', data=np.array(feature_names, dtype='S'))
        f['uns'].attrs['source'] = 'SecActPy'

        # Add anndata compatibility attributes (minimal, for basic reading)
        # Note: For full compatibility, use anndata to write the file
        f.attrs['encoding-type'] = 'anndata'
        f.attrs['encoding-version'] = '0.1.0'

        # obs attributes
        f['obs'].attrs['encoding-type'] = 'dataframe'
        f['obs'].attrs['encoding-version'] = '0.2.0'
        f['obs'].attrs['_index'] = '_index'
        f['obs'].attrs['column-order'] = np.array([], dtype='S')

        # var attributes
        f['var'].attrs['encoding-type'] = 'dataframe'
        f['var'].attrs['encoding-version'] = '0.2.0'
        f['var'].attrs['_index'] = '_index'
        f['var'].attrs['column-order'] = np.array([], dtype='S')

    if verbose:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        print(f"  (Note: h5py fallback used. For full compatibility, install anndata)")
        print(f"\nPython usage:")
        print(f"  import anndata")
        print(f"  adata = anndata.read_h5ad('{path.name}')")
        print(f"  beta = adata.X  # ({n_samples} × {n_features})")
        print(f"  zscore = adata.obsm['zscore']")


# =============================================================================
# Utility Functions
# =============================================================================

def get_file_info(path: Union[str, Path]) -> dict[str, Any]:
    """
    Get information about an HDF5 results file without loading data.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.

    Returns
    -------
    dict
        File information including:
        - shape: (n_features, n_samples)
        - datasets: list of dataset names
        - compression: compression used
        - file_size_mb: file size in MB
        - has_feature_names: bool
        - has_sample_names: bool
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py required")

    path = Path(path)
    info = {
        'path': str(path),
        'file_size_mb': path.stat().st_size / (1024 * 1024)
    }

    with h5py.File(path, 'r') as f:
        info['datasets'] = list(f.keys())

        if 'beta' in f:
            info['shape'] = f['beta'].shape
            info['dtype'] = str(f['beta'].dtype)
            info['compression'] = f['beta'].compression
            info['chunks'] = f['beta'].chunks

        info['has_feature_names'] = 'feature_names' in f.attrs
        info['has_sample_names'] = 'sample_names' in f.attrs

        if info['has_feature_names']:
            info['n_features'] = len(f.attrs['feature_names'])
        if info['has_sample_names']:
            info['n_samples'] = len(f.attrs['sample_names'])

    return info


def concatenate_results(
    result_files: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    axis: int = 1
) -> Optional[dict[str, Any]]:
    """
    Concatenate multiple result files.

    Useful for combining results from parallel batch processing.

    Parameters
    ----------
    result_files : list
        List of HDF5 file paths to concatenate.
    output_path : str or Path, optional
        If provided, save concatenated results to this path.
        Otherwise return in memory.
    axis : int, default=1
        Axis to concatenate along (1=samples, 0=features).

    Returns
    -------
    dict or None
        Concatenated results if output_path is None.
    """
    if not result_files:
        raise ValueError("No files to concatenate")

    # Load all results
    all_results = [load_results(f, load_arrays=True) for f in result_files]

    # Concatenate arrays
    concatenated = {}
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        arrays = [r[name] for r in all_results if name in r]
        if arrays:
            concatenated[name] = np.concatenate(arrays, axis=axis)

    # Concatenate names
    if axis == 1:  # Concatenating samples
        all_sample_names = []
        for r in all_results:
            if 'sample_names' in r:
                all_sample_names.extend(r['sample_names'])
        if all_sample_names:
            concatenated['sample_names'] = all_sample_names

        # Feature names should be the same
        if 'feature_names' in all_results[0]:
            concatenated['feature_names'] = all_results[0]['feature_names']
    else:  # Concatenating features
        all_feature_names = []
        for r in all_results:
            if 'feature_names' in r:
                all_feature_names.extend(r['feature_names'])
        if all_feature_names:
            concatenated['feature_names'] = all_feature_names

        if 'sample_names' in all_results[0]:
            concatenated['sample_names'] = all_results[0]['sample_names']

    if output_path is not None:
        save_results(concatenated, output_path)
        return None

    return concatenated


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("SecActPy IO Module - Testing")
    print("=" * 60)

    print("\nDependencies:")
    print(f"  h5py available: {H5PY_AVAILABLE}")
    print(f"  anndata available: {ANNDATA_AVAILABLE}")

    if not H5PY_AVAILABLE:
        print("\nSkipping tests - h5py not installed")
        exit(0)

    np.random.seed(42)

    # Create test results
    n_features = 10
    n_samples = 100

    results = {
        'beta': np.random.randn(n_features, n_samples),
        'se': np.abs(np.random.randn(n_features, n_samples)),
        'zscore': np.random.randn(n_features, n_samples),
        'pvalue': np.random.uniform(0, 1, (n_features, n_samples)),
        'feature_names': [f"Protein_{i}" for i in range(n_features)],
        'sample_names': [f"Sample_{i}" for i in range(n_samples)]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Save and load HDF5
        print("\n1. Testing HDF5 save/load...")
        h5_path = os.path.join(tmpdir, "test_results.h5")
        save_results(results, h5_path)

        loaded = load_results(h5_path)

        if np.allclose(loaded['beta'], results['beta']):
            print("   ✓ HDF5 round-trip successful")
        else:
            print("   ✗ HDF5 data mismatch!")

        if loaded['feature_names'] == results['feature_names']:
            print("   ✓ Feature names preserved")
        else:
            print("   ✗ Feature names mismatch!")

        # Test 2: File info
        print("\n2. Testing file info...")
        info = get_file_info(h5_path)
        print(f"   Shape: {info['shape']}")
        print(f"   Datasets: {info['datasets']}")
        print(f"   Size: {info['file_size_mb']:.3f} MB")

        # Test 3: Convert to DataFrames
        print("\n3. Testing DataFrame conversion...")
        dfs = results_to_dataframes(loaded)
        print(f"   Beta DataFrame shape: {dfs['beta'].shape}")
        print(f"   Index: {dfs['beta'].index[:3].tolist()}...")
        print(f"   Columns: {dfs['beta'].columns[:3].tolist()}...")

        # Test 4: AnnData conversion
        if ANNDATA_AVAILABLE:
            print("\n4. Testing AnnData conversion...")
            adata = results_to_anndata(loaded)
            print(f"   AnnData shape: {adata.shape}")
            print(f"   Layers: {list(adata.layers.keys())}")
            print(f"   obs (samples): {adata.obs.index[:3].tolist()}...")
            print(f"   var (features): {adata.var.index[:3].tolist()}...")

            # Test direct loading
            adata2 = load_as_anndata(h5_path)
            if adata2.shape == adata.shape:
                print("   ✓ Direct AnnData loading works")
        else:
            print("\n4. Skipping AnnData test (not installed)")

        # Test 5: CSV export
        print("\n5. Testing CSV export...")
        csv_path = os.path.join(tmpdir, "test_results.csv")
        save_results(results, csv_path, format='csv')

        # Check files exist
        csv_files = [f for f in os.listdir(tmpdir) if f.endswith('.csv')]
        print(f"   Created {len(csv_files)} CSV files: {csv_files[:2]}...")

        # Test 6: Lazy loading
        print("\n6. Testing lazy loading...")
        lazy_results = load_results(h5_path, load_arrays=False)
        print(f"   Beta type: {type(lazy_results['beta'])}")

        # Access a slice
        slice_data = lazy_results['beta'][:, :10]
        print(f"   Slice shape: {slice_data.shape}")

        # Close file handle
        lazy_results['_file'].close()
        print("   ✓ Lazy loading works")

        # Test 7: Concatenation
        print("\n7. Testing concatenation...")
        # Create two result files
        results1 = {k: v[:, :50] if isinstance(v, np.ndarray) else v[:50] if isinstance(v, list) and k == 'sample_names' else v
                    for k, v in results.items()}
        results2 = {k: v[:, 50:] if isinstance(v, np.ndarray) else v[50:] if isinstance(v, list) and k == 'sample_names' else v
                    for k, v in results.items()}

        path1 = os.path.join(tmpdir, "results_part1.h5")
        path2 = os.path.join(tmpdir, "results_part2.h5")
        save_results(results1, path1)
        save_results(results2, path2)

        concatenated = concatenate_results([path1, path2])
        print(f"   Concatenated shape: {concatenated['beta'].shape}")

        if concatenated['beta'].shape == (n_features, n_samples):
            print("   ✓ Concatenation works")
        else:
            print("   ✗ Concatenation shape mismatch!")

    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
