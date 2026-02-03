"""
Built-in signature matrices for SecActPy.

This module provides functions to load bundled signature matrices
for secreted protein activity inference.

Available Signatures:
---------------------
- SecAct: Secreted protein activity signature (default)
- CytoSig: Cytokine signaling signature

Usage:
------
    >>> from secactpy import load_signature
    >>>
    >>> # Load default (SecAct)
    >>> sig = load_signature()
    >>>
    >>> # Load specific signature
    >>> sig = load_signature("cytosig")
    >>>
    >>> # Or use convenience functions
    >>> from secactpy.signatures import load_secact, load_cytosig
    >>> secact_sig = load_secact()
    >>> cytosig_sig = load_cytosig()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import warnings

try:
    from importlib.resources import files, as_file
    IMPORTLIB_AVAILABLE = True
except ImportError:
    # Python < 3.9 fallback
    try:
        from importlib_resources import files, as_file
        IMPORTLIB_AVAILABLE = True
    except ImportError:
        IMPORTLIB_AVAILABLE = False

__all__ = [
    'load_signature',
    'load_secact',
    'load_cytosig',
    'load_lincytosig',
    'list_signatures',
    'get_signature_info',
    'AVAILABLE_SIGNATURES',
]


# =============================================================================
# Constants
# =============================================================================

AVAILABLE_SIGNATURES = {
    'secact': {
        'filename': 'SecAct.tsv.gz',
        'description': 'Secreted protein activity signature',
        'reference': 'SecAct package',
    },
    'cytosig': {
        'filename': 'CytoSig.tsv.gz',
        'description': 'Cytokine signaling signature',
        'reference': 'CytoSig database',
    },
    'lincytosig': {
        'filename': 'LinCytoSig.csv',
        'description': 'Cell-type-specific cytokine signatures (median-aggregated from CytoSig DE database)',
        'reference': 'CytoSig database, cell-type stratified',
        'external': True,  # Not bundled with package
    },
}

# Default path for LinCytoSig (external signature)
LINCYTOSIG_DEFAULT_PATH = Path('/data/parks34/projects/2secactpy/results/celltype_signatures/celltype_cytokine_signatures.csv')

DEFAULT_SIGNATURE = 'secact'


# =============================================================================
# Data Path Resolution
# =============================================================================

def _get_data_path() -> Path:
    """Get path to the data directory."""
    # Method 1: importlib.resources (recommended for installed packages)
    if IMPORTLIB_AVAILABLE:
        try:
            data_files = files('secactpy') / 'data'
            return data_files
        except (TypeError, ModuleNotFoundError):
            pass

    # Method 2: Relative path from this file (for development)
    module_path = Path(__file__).parent
    data_path = module_path / 'data'

    if data_path.exists():
        return data_path

    raise FileNotFoundError(
        "Cannot locate secactpy data directory. "
        "Ensure the package is properly installed."
    )


def _load_tsv_gz(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a tsv.gz signature matrix file.

    Handles the format where the first column (gene names) has no header.
    """
    # Read with first column as index
    df = pd.read_csv(
        filepath,
        sep='\t',
        index_col=0,
        compression='gzip'
    )

    # Clean up index name (might be empty or None)
    df.index.name = 'gene'

    # Ensure float dtype
    df = df.astype(np.float64)

    # Handle any NaN values
    if df.isna().any().any():
        n_nan = df.isna().sum().sum()
        warnings.warn(f"Signature contains {n_nan} NaN values. Filling with 0.")
        df = df.fillna(0.0)

    return df


# =============================================================================
# Main Loading Functions
# =============================================================================

def load_signature(
    name: str = DEFAULT_SIGNATURE,
    features: Optional[List[str]] = None,
    genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load a bundled signature matrix.

    Parameters
    ----------
    name : str, default='secact'
        Signature name. Use `list_signatures()` to see available options.
        Case-insensitive.
    features : list, optional
        Subset of features (columns) to load. If None, loads all.
    genes : list, optional
        Subset of genes (rows) to load. If None, loads all.

    Returns
    -------
    DataFrame
        Signature matrix with genes as rows and features as columns.
        Shape: (n_genes, n_features)

    Examples
    --------
    >>> sig = load_signature()  # Load SecAct (default)
    >>> print(sig.shape)
    (12000, 150)

    >>> sig = load_signature('cytosig')  # Load CytoSig
    >>> print(sig.columns[:5].tolist())
    ['Activin A', 'BDNF', 'BMP2', 'BMP4', 'BMP6']

    >>> # Load subset
    >>> sig = load_signature('secact', features=['IL6', 'TNF', 'IFNG'])

    Raises
    ------
    ValueError
        If signature name is not recognized.
    FileNotFoundError
        If signature file is not found.
    """
    name_lower = name.lower()

    if name_lower not in AVAILABLE_SIGNATURES:
        available = ', '.join(AVAILABLE_SIGNATURES.keys())
        raise ValueError(
            f"Unknown signature '{name}'. Available: {available}"
        )

    sig_info = AVAILABLE_SIGNATURES[name_lower]

    # Handle external signatures (not bundled with package)
    if sig_info.get('external', False):
        if name_lower == 'lincytosig':
            return load_lincytosig(features=features, genes=genes)
        else:
            raise ValueError(f"External signature '{name}' requires custom loader")

    filename = sig_info['filename']

    # Get file path
    data_path = _get_data_path()

    # Handle both Path and Traversable (from importlib.resources)
    if IMPORTLIB_AVAILABLE and not isinstance(data_path, Path):
        # Using importlib.resources
        file_ref = data_path / filename
        with as_file(file_ref) as filepath:
            df = _load_tsv_gz(filepath)
    else:
        # Using direct path
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Signature file not found: {filepath}. "
                f"Ensure {filename} is in secactpy/data/"
            )
        df = _load_tsv_gz(filepath)

    # Subset features if requested
    if features is not None:
        missing = set(features) - set(df.columns)
        if missing:
            warnings.warn(f"Features not found in signature: {missing}")
        available_features = [f for f in features if f in df.columns]
        df = df[available_features]

    # Subset genes if requested
    if genes is not None:
        # Convert to string for matching
        df.index = df.index.astype(str)
        genes_str = [str(g) for g in genes]
        available_genes = [g for g in genes_str if g in df.index]
        if len(available_genes) < len(genes):
            n_missing = len(genes) - len(available_genes)
            warnings.warn(f"{n_missing} requested genes not found in signature")
        df = df.loc[available_genes]

    return df


def load_secact(
    features: Optional[List[str]] = None,
    genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load the SecAct signature matrix.

    Convenience function for `load_signature('secact')`.

    Parameters
    ----------
    features : list, optional
        Subset of features (proteins) to load.
    genes : list, optional
        Subset of genes to load.

    Returns
    -------
    DataFrame
        SecAct signature matrix.
    """
    return load_signature('secact', features=features, genes=genes)


def load_cytosig(
    features: Optional[List[str]] = None,
    genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load the CytoSig signature matrix.

    Convenience function for `load_signature('cytosig')`.

    Parameters
    ----------
    features : list, optional
        Subset of features (cytokines) to load.
    genes : list, optional
        Subset of genes to load.

    Returns
    -------
    DataFrame
        CytoSig signature matrix.
    """
    return load_signature('cytosig', features=features, genes=genes)


def load_lincytosig(
    path: Optional[Union[str, Path]] = None,
    features: Optional[List[str]] = None,
    genes: Optional[List[str]] = None,
    cell_types: Optional[List[str]] = None,
    cytokines: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load the LinCytoSig signature matrix.

    LinCytoSig contains cell-type-specific cytokine response signatures,
    derived from the CytoSig differential expression database using
    median aggregation across experiments.

    Signature columns are named as "CellType__Cytokine" (e.g., "Macrophage__IFNG").

    Parameters
    ----------
    path : str or Path, optional
        Path to LinCytoSig CSV file. If None, uses default location.
    features : list, optional
        Subset of features (CellType__Cytokine columns) to load.
    genes : list, optional
        Subset of genes to load.
    cell_types : list, optional
        Filter to specific cell types (e.g., ["Macrophage", "T_CD4"]).
    cytokines : list, optional
        Filter to specific cytokines (e.g., ["IFNG", "TNFA"]).

    Returns
    -------
    DataFrame
        LinCytoSig signature matrix (genes x cell_type__cytokine).
        Shape: (19918, 178) for full matrix.

    Examples
    --------
    >>> sig = load_lincytosig()
    >>> print(sig.shape)
    (19918, 178)

    >>> # Filter to macrophage signatures only
    >>> mac_sig = load_lincytosig(cell_types=["Macrophage"])
    >>> print(mac_sig.columns.tolist())
    ['Macrophage__IFNG', 'Macrophage__TNFA', ...]

    >>> # Filter to IFNG responses across all cell types
    >>> ifng_sig = load_lincytosig(cytokines=["IFNG"])
    """
    # Determine path
    if path is None:
        path = LINCYTOSIG_DEFAULT_PATH
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"LinCytoSig file not found: {path}\n"
            "Generate it using: python scripts/build_celltype_signatures.py"
        )

    # Load the signature matrix
    df = pd.read_csv(path, index_col=0)
    df.index.name = 'gene'

    # Ensure float dtype
    df = df.astype(np.float64)

    # Handle NaN values (sparse matrix may have NaNs for untested combinations)
    if df.isna().any().any():
        n_nan = df.isna().sum().sum()
        warnings.warn(f"LinCytoSig contains {n_nan} NaN values. Filling with 0.")
        df = df.fillna(0.0)

    # Filter by cell types
    if cell_types is not None:
        matching_cols = [
            col for col in df.columns
            if col.split('__')[0] in cell_types
        ]
        if not matching_cols:
            available_cts = sorted(set(col.split('__')[0] for col in df.columns))
            raise ValueError(
                f"No columns match cell_types {cell_types}. "
                f"Available: {available_cts[:10]}..."
            )
        df = df[matching_cols]

    # Filter by cytokines
    if cytokines is not None:
        matching_cols = [
            col for col in df.columns
            if col.split('__')[1] in cytokines
        ]
        if not matching_cols:
            available_cyto = sorted(set(col.split('__')[1] for col in df.columns))
            raise ValueError(
                f"No columns match cytokines {cytokines}. "
                f"Available: {available_cyto[:10]}..."
            )
        df = df[matching_cols]

    # Filter by specific features (overrides cell_types/cytokines if both given)
    if features is not None:
        missing = set(features) - set(df.columns)
        if missing:
            warnings.warn(f"Features not found in LinCytoSig: {list(missing)[:5]}...")
        available_features = [f for f in features if f in df.columns]
        df = df[available_features]

    # Filter by genes
    if genes is not None:
        df.index = df.index.astype(str)
        genes_str = [str(g) for g in genes]
        available_genes = [g for g in genes_str if g in df.index]
        if len(available_genes) < len(genes):
            n_missing = len(genes) - len(available_genes)
            warnings.warn(f"{n_missing} requested genes not found in LinCytoSig")
        df = df.loc[available_genes]

    return df


# =============================================================================
# Information Functions
# =============================================================================

def list_signatures() -> List[str]:
    """
    List available signature matrices.

    Returns
    -------
    list
        Names of available signatures.

    Examples
    --------
    >>> print(list_signatures())
    ['secact', 'cytosig']
    """
    return list(AVAILABLE_SIGNATURES.keys())


def get_signature_info(name: Optional[str] = None) -> dict:
    """
    Get information about signature matrix(es).

    Parameters
    ----------
    name : str, optional
        Signature name. If None, returns info for all signatures.

    Returns
    -------
    dict
        Signature information including description, shape (if loaded),
        and reference.

    Examples
    --------
    >>> info = get_signature_info('secact')
    >>> print(info['description'])
    'Secreted protein activity signature'

    >>> all_info = get_signature_info()
    >>> for name, info in all_info.items():
    ...     print(f"{name}: {info['description']}")
    """
    if name is not None:
        name_lower = name.lower()
        if name_lower not in AVAILABLE_SIGNATURES:
            raise ValueError(f"Unknown signature: {name}")

        info = AVAILABLE_SIGNATURES[name_lower].copy()

        # Try to load and get shape
        try:
            sig = load_signature(name_lower)
            info['n_genes'] = sig.shape[0]
            info['n_features'] = sig.shape[1]
            info['features'] = sig.columns.tolist()[:10]  # First 10
            info['genes_sample'] = sig.index.tolist()[:5]  # First 5
        except Exception as e:
            info['load_error'] = str(e)

        return info

    # Return info for all signatures
    all_info = {}
    for sig_name in AVAILABLE_SIGNATURES:
        all_info[sig_name] = get_signature_info(sig_name)

    return all_info


# =============================================================================
# Validation Functions
# =============================================================================

def validate_signature(sig: pd.DataFrame) -> dict:
    """
    Validate a signature matrix.

    Checks for common issues like NaN values, zero variance features,
    and duplicate genes.

    Parameters
    ----------
    sig : DataFrame
        Signature matrix to validate.

    Returns
    -------
    dict
        Validation results with 'valid' bool and any 'warnings' or 'errors'.
    """
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {
            'n_genes': sig.shape[0],
            'n_features': sig.shape[1],
        }
    }

    # Check for NaN
    n_nan = sig.isna().sum().sum()
    if n_nan > 0:
        result['warnings'].append(f"Contains {n_nan} NaN values")

    # Check for zero variance features
    zero_var = (sig.std() == 0).sum()
    if zero_var > 0:
        result['warnings'].append(f"{zero_var} features have zero variance")

    # Check for duplicate genes
    n_dup = sig.index.duplicated().sum()
    if n_dup > 0:
        result['warnings'].append(f"{n_dup} duplicate gene names")

    # Check for empty rows/columns
    empty_rows = (sig == 0).all(axis=1).sum()
    empty_cols = (sig == 0).all(axis=0).sum()
    if empty_rows > 0:
        result['warnings'].append(f"{empty_rows} genes have all zeros")
    if empty_cols > 0:
        result['warnings'].append(f"{empty_cols} features have all zeros")

    # Overall validity
    if result['errors']:
        result['valid'] = False

    return result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SecActPy Signatures Module - Testing")
    print("=" * 60)

    # Test 1: List signatures
    print("\n1. Available signatures:")
    for name in list_signatures():
        print(f"   - {name}")

    # Test 2: Load signatures
    print("\n2. Loading signatures...")

    for name in list_signatures():
        try:
            sig = load_signature(name)
            print(f"   {name}: {sig.shape[0]} genes × {sig.shape[1]} features")
            print(f"      Features: {sig.columns[:5].tolist()}...")
            print(f"      Genes: {sig.index[:3].tolist()}...")

            # Validate
            validation = validate_signature(sig)
            if validation['warnings']:
                for w in validation['warnings']:
                    print(f"      Warning: {w}")
            else:
                print("      ✓ Validation passed")

        except FileNotFoundError:
            print(f"   {name}: File not found (expected during development)")
        except Exception as e:
            print(f"   {name}: Error - {e}")

    # Test 3: Convenience functions
    print("\n3. Testing convenience functions...")
    try:
        secact = load_secact()
        print(f"   load_secact(): {secact.shape}")
    except FileNotFoundError:
        print("   load_secact(): File not found (add SecAct.tsv.gz to data/)")

    try:
        cytosig = load_cytosig()
        print(f"   load_cytosig(): {cytosig.shape}")
    except FileNotFoundError:
        print("   load_cytosig(): File not found (add CytoSig.tsv.gz to data/)")

    # Test 4: Signature info
    print("\n4. Signature info:")
    for name in list_signatures():
        try:
            info = get_signature_info(name)
            if 'n_genes' in info:
                print(f"   {name}: {info['n_genes']} genes, {info['n_features']} features")
            else:
                print(f"   {name}: {info.get('load_error', 'Info unavailable')}")
        except Exception as e:
            print(f"   {name}: Error - {e}")

    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
