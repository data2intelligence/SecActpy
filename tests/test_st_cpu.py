#!/usr/bin/env python3
"""
Test Script: CPU Spatial Transcriptomics Inference Validation

Validates SecActPy ST inference against RidgeR output.
Tests Visium data by default, can also test CosMx with --cosmx flag.

Dataset:
    - Visium_HCC (10X Visium hepatocellular carcinoma)
    - LIHC_CosMx_data.h5ad (single-cell resolution ST)

Reference: R output from RidgeR::SecAct.activity.inference.ST

Usage:
    python tests/test_st_cpu.py              # Visium dataset
    python tests/test_st_cpu.py --cosmx      # CosMx dataset
    python tests/test_st_cpu.py --input data.h5ad --reference ref.h5ad
    python tests/test_st_cpu.py --save

For faster R comparison, convert R text outputs to h5ad first:
    Rscript scripts/save_results_h5ad.R dataset/output/signature/CosMx

Expected output:
    All arrays should match R output exactly (or within numerical tolerance).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time

# Add package to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy import secact_activity_inference_st, load_visium_10x, load_signature


# =============================================================================
# Configuration (defaults)
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"

# Visium configuration (defaults)
DEFAULT_VISIUM_INPUT = DATA_DIR / "input" / "Visium_HCC"
DEFAULT_VISIUM_REFERENCE = DATA_DIR / "output" / "signature" / "ST"
VISIUM_MIN_GENES = 1000
VISIUM_SCALE_FACTOR = 1e5

# CosMx configuration (defaults)
DEFAULT_COSMX_INPUT = DATA_DIR / "input" / "LIHC_CosMx_data.h5ad"
DEFAULT_COSMX_REFERENCE = DATA_DIR / "output" / "signature" / "CosMx"
COSMX_MIN_GENES = 50
COSMX_SCALE_FACTOR = 1000

# Common parameters
LAMBDA = 5e5
NRAND = 1000
SEED = 0
GROUP_COR = 0.9


# =============================================================================
# R Output Loading Functions
# =============================================================================

def load_r_output_h5ad(output_dir: Path) -> dict:
    """
    Load R output from h5ad file (fast).

    Supports two formats:
    1. New format: X=beta, obsm/{se,zscore,pvalue}
    2. Old format: obsm/{beta,se,zscore,pvalue}

    Returns dict with 'beta', 'se', 'zscore', 'pvalue' as DataFrames (proteins x cells).
    """
    # Try both possible filenames
    h5ad_path = output_dir / "output.h5ad"
    if not h5ad_path.exists():
        h5ad_path = output_dir / "results.h5ad"
    if not h5ad_path.exists():
        return None

    print(f"  Loading from h5ad: {h5ad_path}")
    start = time.time()

    # Try anndata first (works for standard h5ad files)
    try:
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path)

        protein_names = list(adata.var_names)
        cell_names = list(adata.obs_names)

        print(f"    Proteins: {len(protein_names)}, Cells: {len(cell_names)}")

        result = {}

        # X is beta (cells x proteins), transpose to (proteins x cells)
        if adata.X is not None:
            if hasattr(adata.X, 'toarray'):
                X = adata.X.toarray()
            else:
                X = np.array(adata.X)
            result['beta'] = pd.DataFrame(
                X.T,
                index=protein_names,
                columns=cell_names
            )
            print(f"    beta: {result['beta'].shape}")

        # Other matrices from obsm
        for name in ['se', 'zscore', 'pvalue']:
            if name in adata.obsm:
                mat = adata.obsm[name]
                if hasattr(mat, 'toarray'):
                    mat = mat.toarray()
                result[name] = pd.DataFrame(
                    mat.T,
                    index=protein_names,
                    columns=cell_names
                )
                print(f"    {name}: {result[name].shape}")

        elapsed = time.time() - start
        print(f"  Loaded in {elapsed:.1f}s")

        return result if result else None

    except Exception as e:
        print(f"  anndata failed: {e}")
        print("  Trying h5py fallback...")

    # Fallback to h5py for non-standard structures
    import h5py

    try:
        with h5py.File(h5ad_path, 'r') as f:
            # Get names from both obs and var
            obs_names = None
            var_names = None
            
            if 'obs/_index' in f:
                obs_names = [x.decode() if isinstance(x, bytes) else x
                            for x in f['obs/_index'][:]]
            if 'var/_index' in f:
                var_names = [x.decode() if isinstance(x, bytes) else x
                            for x in f['var/_index'][:]]
            
            if obs_names is None or var_names is None:
                print("  Warning: Could not find obs/_index or var/_index")
                return None
            
            # Determine which is proteins and which is cells
            # Standard format: obs=cells, var=proteins
            # But some R outputs have them swapped
            # Use heuristic: proteins are typically fewer than cells for ST data
            n_obs = len(obs_names)
            n_var = len(var_names)
            
            # Check if dimensions seem swapped (var has more entries than obs)
            # This happens when R saves with obs=proteins, var=cells
            swapped = False
            if n_var > n_obs and n_var > 10000:
                # Likely swapped: var has cells, obs has proteins
                print(f"    Detected swapped dimensions: obs={n_obs}, var={n_var}")
                print(f"    Interpreting as: proteins={n_obs}, cells={n_var}")
                protein_names = obs_names
                cell_names = var_names
                swapped = True
            else:
                # Standard format
                protein_names = var_names
                cell_names = obs_names

            print(f"    Proteins: {len(protein_names)}, Cells: {len(cell_names)}")

            result = {}

            # Try X for beta first (new format)
            if 'X' in f:
                # Check if X is a group (sparse matrix) or dataset
                if isinstance(f['X'], h5py.Group):
                    # Sparse matrix - reconstruct
                    try:
                        from scipy import sparse
                        data = f['X/data'][:]
                        indices = f['X/indices'][:]
                        indptr = f['X/indptr'][:]
                        shape = tuple(f['X'].attrs.get('shape', (n_obs, n_var)))
                        arr = sparse.csr_matrix((data, indices, indptr), shape=shape).toarray()
                    except Exception as e:
                        print(f"    Warning: Could not load sparse X: {e}")
                        arr = None
                else:
                    arr = f['X'][:]
                
                if arr is not None:
                    # Ensure shape is (proteins, cells)
                    if swapped:
                        # If swapped, X is (proteins, cells) - no transpose needed
                        if arr.shape == (len(protein_names), len(cell_names)):
                            pass  # Already correct
                        else:
                            arr = arr.T
                    else:
                        # Standard: X is (cells, proteins), transpose to (proteins, cells)
                        if arr.shape == (len(cell_names), len(protein_names)):
                            arr = arr.T
                    
                    result['beta'] = pd.DataFrame(arr, index=protein_names, columns=cell_names)
                    print(f"    beta: {result['beta'].shape}")

            # Other matrices from obsm (or beta if X wasn't found)
            for name in ['beta', 'se', 'zscore', 'pvalue']:
                if name == 'beta' and 'beta' in result:
                    continue  # Already loaded from X

                key = f'obsm/{name}'
                if key in f:
                    # Check if sparse
                    if isinstance(f[key], h5py.Group):
                        try:
                            from scipy import sparse
                            data = f[f'{key}/data'][:]
                            indices = f[f'{key}/indices'][:]
                            indptr = f[f'{key}/indptr'][:]
                            shape = tuple(f[key].attrs.get('shape', (n_obs, n_var)))
                            arr = sparse.csr_matrix((data, indices, indptr), shape=shape).toarray()
                        except Exception as e:
                            print(f"    Warning: Could not load sparse {name}: {e}")
                            continue
                    else:
                        arr = f[key][:]
                    
                    # Ensure shape is (proteins, cells)
                    if swapped:
                        # If swapped, obsm is stored as (proteins, cells)
                        if arr.shape == (len(protein_names), len(cell_names)):
                            pass
                        else:
                            arr = arr.T
                    else:
                        # Standard: obsm is (cells, proteins), transpose
                        if arr.shape == (len(cell_names), len(protein_names)):
                            arr = arr.T
                    
                    result[name] = pd.DataFrame(arr, index=protein_names, columns=cell_names)
                    print(f"    {name}: {result[name].shape}")

        elapsed = time.time() - start
        print(f"  Loaded in {elapsed:.1f}s")
        return result if result else None

    except Exception as e2:
        print(f"  h5py fallback also failed: {e2}")
        import traceback
        traceback.print_exc()
        print("  Falling back to text files...")
        return None


def load_r_output_txt(output_dir: Path) -> dict:
    """Load R output from text files (slow for large files)."""
    result = {}

    for name in ['beta', 'se', 'zscore', 'pvalue']:
        filepath = output_dir / f"{name}.txt"
        if filepath.exists():
            print(f"  Loading {name}...", end=" ", flush=True)
            start = time.time()
            df = pd.read_csv(filepath, sep=r'\s+', index_col=0)
            elapsed = time.time() - start
            result[name] = df
            print(f"{df.shape} in {elapsed:.1f}s")
        else:
            print(f"  Warning: {name}.txt not found")

    return result


def load_r_output(output_dir: Path) -> dict:
    """
    Load R output files, preferring h5ad format if available.

    To create h5ad from R outputs:
        Rscript scripts/save_results_h5ad.R <output_dir>
    """
    # Try h5ad first (much faster)
    result = load_r_output_h5ad(output_dir)
    if result:
        return result

    # Fall back to text files
    print("  H5AD not found, loading text files (slower)...")
    print("  Tip: Run 'Rscript scripts/save_results_h5ad.R <output_dir>' for faster loading")
    return load_r_output_txt(output_dir)


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_results(py_result: dict, r_result: dict, tolerance: float = 1e-10) -> dict:
    """Compare Python and R results."""
    comparison = {}

    for name in ['beta', 'se', 'zscore', 'pvalue']:
        if name not in py_result or name not in r_result:
            comparison[name] = {'status': 'MISSING', 'message': 'Array not found'}
            continue

        py_arr = py_result[name]
        r_arr = r_result[name]

        # Find common rows and columns
        common_rows = py_arr.index.intersection(r_arr.index)
        common_cols = py_arr.columns.intersection(r_arr.columns)

        if len(common_rows) == 0 or len(common_cols) == 0:
            comparison[name] = {
                'status': 'FAIL',
                'message': f'No common rows/cols. Python: {py_arr.shape}, R: {r_arr.shape}'
            }
            continue

        # Report shape differences
        if py_arr.shape != r_arr.shape:
            print(f"    Note: {name} shape differs - Python {py_arr.shape} vs R {r_arr.shape}")
            print(f"    Comparing {len(common_rows)} common proteins × {len(common_cols)} common cells")

        # Align by row and column names
        py_aligned = py_arr.loc[common_rows, common_cols]
        r_aligned = r_arr.loc[common_rows, common_cols]

        # Calculate difference
        diff = np.abs(py_aligned.values - r_aligned.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        n_compared = len(common_rows) * len(common_cols)
        
        # Calculate correlation
        py_flat = py_aligned.values.flatten()
        r_flat = r_aligned.values.flatten()
        valid = ~(np.isnan(py_flat) | np.isnan(r_flat))
        if valid.sum() > 1:
            corr = np.corrcoef(py_flat[valid], r_flat[valid])[0, 1]
        else:
            corr = np.nan

        if max_diff <= tolerance:
            comparison[name] = {
                'status': 'PASS',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'correlation': corr,
                'n_compared': n_compared,
                'message': f'Max diff: {max_diff:.2e}, Corr: {corr:.10f}'
            }
        else:
            # Find location of max diff
            max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
            max_row = common_rows[max_idx[0]]
            max_col = common_cols[max_idx[1]]

            comparison[name] = {
                'status': 'FAIL',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'correlation': corr,
                'n_compared': n_compared,
                'max_loc': (max_row, max_col),
                'message': f'Max diff: {max_diff:.2e}, Corr: {corr:.10f}'
            }

    return comparison


def load_cosmx_data(input_file, min_genes=50, verbose=True):
    """Load CosMx data from h5ad file."""
    import anndata as ad
    from scipy import sparse

    adata = ad.read_h5ad(input_file)

    # Always use adata.X and adata.var_names (not .raw)
    # The .raw layer may have different gene IDs
    counts_matrix = adata.X
    gene_names = list(adata.var_names)
    cell_names = list(adata.obs_names)

    if verbose:
        print(f"   Using adata.X (shape: {counts_matrix.shape})")
        print(f"   Total genes: {len(gene_names)}")
        print(f"   Total cells: {len(cell_names)}")
        print(f"   Sample gene names: {gene_names[:10]}")

    # Apply QC filter
    if sparse.issparse(counts_matrix):
        genes_per_cell = np.asarray((counts_matrix > 0).sum(axis=1)).ravel()
    else:
        genes_per_cell = (counts_matrix > 0).sum(axis=1)

    keep_cells = genes_per_cell >= min_genes
    n_before = len(cell_names)
    n_after = keep_cells.sum()

    counts_matrix = counts_matrix[keep_cells, :]
    cell_names = [c for c, k in zip(cell_names, keep_cells) if k]

    if verbose:
        print(f"   Cells after QC (>={min_genes} genes): {n_after} / {n_before}")

    # Transpose to (genes × cells)
    if sparse.issparse(counts_matrix):
        counts_transposed = counts_matrix.T.tocsr()
        counts_df = pd.DataFrame.sparse.from_spmatrix(
            counts_transposed, index=gene_names, columns=cell_names
        ).sparse.to_dense()
    else:
        counts_df = pd.DataFrame(counts_matrix.T, index=gene_names, columns=cell_names)

    if verbose:
        print(f"   Final shape: {counts_df.shape} (genes × cells)")

    return counts_df


# =============================================================================
# Main Test
# =============================================================================

def main(input_file=None, reference=None, cosmx=False, save_output=False):
    """
    Run spatial transcriptomics inference validation.

    Parameters
    ----------
    input_file : str, optional
        Path to input file. If None, uses default for platform.
    reference : str, optional
        Path to reference output (H5AD file or folder with TXT files).
        If None, uses default reference directory.
    cosmx : bool, default=False
        If True, use CosMx dataset. If False, use Visium dataset.
    save_output : bool, default=False
        If True, save results to files.
    """
    platform = "CosMx" if cosmx else "Visium"

    print("=" * 70)
    print(f"SecActPy Spatial Transcriptomics Validation Test (CPU) - {platform}")
    print("=" * 70)

    # Set configuration based on platform
    if cosmx:
        default_input = DEFAULT_COSMX_INPUT
        default_reference = DEFAULT_COSMX_REFERENCE
        min_genes = COSMX_MIN_GENES
        scale_factor = COSMX_SCALE_FACTOR
        sig_filter = True
    else:
        default_input = DEFAULT_VISIUM_INPUT
        default_reference = DEFAULT_VISIUM_REFERENCE
        min_genes = VISIUM_MIN_GENES
        scale_factor = VISIUM_SCALE_FACTOR
        sig_filter = False

    # Determine input path
    input_path = Path(input_file) if input_file is not None else default_input
    
    # Determine reference path
    reference_path = Path(reference) if reference is not None else default_reference

    # Check files
    print("\n1. Checking files...")
    if not input_path.exists():
        print(f"   ERROR: Input not found: {input_path}")
        return False
    print(f"   Input: {input_path}")

    validate = True
    
    # Check for reference output (H5AD or directory)
    if reference_path.suffix == '.h5ad':
        h5ad_ref = reference_path
    else:
        h5ad_ref = reference_path / "output.h5ad"
    
    if h5ad_ref.exists():
        print(f"   Reference: {h5ad_ref}")
    elif reference_path.is_dir():
        txt_files = list(reference_path.glob("*.txt"))
        if txt_files:
            print(f"   Reference: {reference_path} (TXT files)")
        else:
            print(f"   Warning: Reference output not found: {reference_path}")
            validate = False
    else:
        print(f"   Warning: Reference output not found: {reference_path}")
        print("   Will run inference but skip validation.")
        validate = False

    # Load data
    print("\n2. Loading data...")

    if cosmx:
        input_data = load_cosmx_data(input_path, min_genes=min_genes, verbose=True)
    else:
        input_data = load_visium_10x(input_path, min_genes=min_genes, verbose=True)
        print(f"   Spots: {len(input_data['spot_names'])}")
        print(f"   Genes: {len(input_data['gene_names'])}")

    # Check gene overlap with signature
    print("\n   Checking gene overlap with signature...")
    sig_df = load_signature("secact")
    sig_genes = set(sig_df.index)

    if isinstance(input_data, pd.DataFrame):
        data_genes = set(input_data.index)
    else:
        data_genes = set(input_data.get('gene_names', []))

    overlap = sig_genes & data_genes
    print(f"   Signature genes: {len(sig_genes)}")
    print(f"   Data genes: {len(data_genes)}")
    print(f"   Overlap: {len(overlap)}")

    if len(overlap) == 0:
        print("\n   WARNING: No direct gene overlap!")
        print(f"   Sample signature genes: {list(sig_genes)[:5]}")
        print(f"   Sample data genes: {list(data_genes)[:5]}")

        # Try case-insensitive matching
        sig_genes_upper = {g.upper(): g for g in sig_genes}
        data_genes_upper = {g.upper(): g for g in data_genes}
        case_overlap = set(sig_genes_upper.keys()) & set(data_genes_upper.keys())

        if len(case_overlap) > 0:
            print(f"\n   Case-insensitive overlap: {len(case_overlap)}")
            print("   The updated inference.py should handle this automatically.")

    # Run inference
    print(f"\n3. Running SecActPy inference (CPU, {platform})...")
    start_time = time.time()

    try:
        py_result = secact_activity_inference_st(
            input_data=input_data,
            scale_factor=scale_factor,
            sig_matrix="secact",
            is_group_sig=True,
            is_group_cor=GROUP_COR,
            lambda_=LAMBDA,
            n_rand=NRAND,
            seed=SEED,
            sig_filter=sig_filter,
            backend="numpy",
            use_gsl_rng=True,  # Use GSL RNG for R compatibility
            use_cache=True,  # Cache permutation tables for faster repeated runs
            verbose=True
        )

        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.2f} seconds")
        print(f"   Result shape: {py_result['beta'].shape}")

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    all_passed = True

    if validate:
        # Load R reference
        print("\n4. Loading R reference output...")
        r_result = load_r_output(reference_path)

        if not r_result:
            print("   Warning: No R output files found!")
            validate = False
        else:
            # Compare
            print("\n5. Comparing results...")
            comparison = compare_results(py_result, r_result)

            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)

            for name, result in comparison.items():
                status = result['status']
                message = result['message']

                if status == 'PASS':
                    print(f"  {name:8s}: ✓ PASS - {message}")
                else:
                    print(f"  {name:8s}: ✗ {status} - {message}")
                    all_passed = False

            print("\n" + "=" * 70)
            if all_passed:
                print("ALL TESTS PASSED! ✓")
                print(f"SecActPy ST ({platform}) produces identical results to RidgeR.")
            else:
                print("SOME TESTS FAILED! ✗")
                print("Check the detailed output above for discrepancies.")
                
                # Save output for CLI comparison
                print("\n--- Saving Python output for CLI comparison ---")
                try:
                    from secactpy.io import save_results_to_h5ad
                    py_output_path = PACKAGE_ROOT / "dataset" / "output" / f"{platform}_py_output.h5ad"
                    save_results_to_h5ad(py_result, py_output_path, verbose=False)
                    print(f"  Saved to: {py_output_path}")
                    print(f"\n  Try CLI comparison:")
                    ref_h5ad = reference_path if reference_path.suffix == '.h5ad' else reference_path / 'output.h5ad'
                    print(f"  secactpy compare {ref_h5ad} {py_output_path}")
                except Exception as e:
                    print(f"  Could not save: {e}")
            print("=" * 70)

    if not validate:
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETE (validation skipped)")
        print("=" * 70)

    # Show sample output
    step_num = 6 if validate else 4
    print(f"\n{step_num}. Sample output (first 5 proteins, first 3 samples):")
    cols = py_result['zscore'].columns[:3]
    print(py_result['zscore'].iloc[:5][cols])

    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            from secactpy.io import save_results_to_h5ad

            output_h5ad = PACKAGE_ROOT / "dataset" / "output" / f"{platform}_cpu_activity.h5ad"
            save_results_to_h5ad(py_result, output_h5ad, verbose=True)
            print(f"   Saved to: {output_h5ad}")

        except Exception as e:
            print(f"   Could not save: {e}")
            import traceback
            traceback.print_exc()

    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy Spatial Transcriptomics Inference (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visium dataset with defaults
  python tests/test_st_cpu.py

  # CosMx dataset
  python tests/test_st_cpu.py --cosmx

  # Custom input and reference
  python tests/test_st_cpu.py --input data.h5ad --reference ref.h5ad
  python tests/test_st_cpu.py -i Visium_folder/ -r ref_folder/

  # Save results
  python tests/test_st_cpu.py --save
  python tests/test_st_cpu.py --cosmx --save
        """
    )
    parser.add_argument(
        '--input', '-i',
        dest='input_file',
        default=None,
        help='Path to input (H5AD file or Visium folder)'
    )
    parser.add_argument(
        '--reference', '-r',
        dest='reference',
        default=None,
        help='Path to R reference output (H5AD file or folder with TXT files)'
    )
    parser.add_argument(
        '--cosmx',
        action='store_true',
        help='Use CosMx dataset (default: Visium)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to h5ad file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(
        input_file=args.input_file,
        reference=args.reference,
        cosmx=args.cosmx,
        save_output=args.save
    )
    sys.exit(0 if success else 1)
