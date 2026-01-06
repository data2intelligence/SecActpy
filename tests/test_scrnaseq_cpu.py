#!/usr/bin/env python3
"""
Test Script: CPU scRNA-seq Inference Validation

Validates SecActPy scRNA-seq inference against RidgeR output.
Tests both cell-type resolution (pseudo-bulk) and single-cell resolution.

Supports both H5AD and TXT reference formats.

Dataset: OV_scRNAseq_CD4.h5ad
Reference: R output from RidgeR::SecAct.activity.inference.scRNAseq

Usage:
    python tests/test_scrnaseq_cpu.py                  # Cell-type resolution
    python tests/test_scrnaseq_cpu.py --single-cell    # Single-cell resolution
    python tests/test_scrnaseq_cpu.py --input data.h5ad --reference ref.h5ad
    python tests/test_scrnaseq_cpu.py --save

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

from secactpy import secact_activity_inference_scrnaseq


# =============================================================================
# Configuration (defaults)
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
DEFAULT_INPUT = DATA_DIR / "input" / "OV_scRNAseq_data.h5ad"

# Default reference output directories
DEFAULT_REFERENCE_CT = DATA_DIR / "output" / "signature" / "scRNAseq_ct_res"
DEFAULT_REFERENCE_SC = DATA_DIR / "output" / "signature" / "scRNAseq_sc_res"

# Parameters matching R defaults
CELL_TYPE_COL = "Annotation"
LAMBDA = 5e5
NRAND = 1000
SEED = 0
GROUP_COR = 0.9


# =============================================================================
# Reference Loading Functions
# =============================================================================

def load_r_output_h5ad(h5ad_path: Path) -> dict:
    """Load R output from H5AD file created by write_secact_to_h5ad()."""
    import anndata as ad
    
    print(f"  Loading H5AD reference: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    
    result = {}
    
    # Get sample and protein names
    # R's write_secact_to_h5ad saves in Python format: obs=samples, var=proteins
    sample_names = list(adata.obs_names)
    protein_names = list(adata.var_names)
    
    print(f"  Shape: {adata.n_obs} samples × {adata.n_vars} proteins")
    
    # X is (samples × proteins), transpose to (proteins × samples) for comparison
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    result['beta'] = pd.DataFrame(X.T, index=protein_names, columns=sample_names)
    print(f"  Loaded beta: {result['beta'].shape}")
    
    # Get obsm matrices (se, zscore, pvalue)
    for name in ['se', 'zscore', 'pvalue']:
        if name in adata.obsm:
            mat = adata.obsm[name]
            if hasattr(mat, 'toarray'):
                mat = mat.toarray()
            else:
                mat = np.array(mat)
            # obsm is (samples × proteins), transpose to (proteins × samples)
            result[name] = pd.DataFrame(mat.T, index=protein_names, columns=sample_names)
            print(f"  Loaded {name}: {result[name].shape}")
        else:
            print(f"  Warning: {name} not found in obsm")
    
    return result


def load_r_output_txt(output_dir: Path) -> dict:
    """Load R output from TXT files."""
    result = {}

    for name in ['beta', 'se', 'zscore', 'pvalue']:
        filepath = output_dir / f"{name}.txt"
        if filepath.exists():
            df = pd.read_csv(filepath, sep=r'\s+', index_col=0)
            result[name] = df
            print(f"  Loaded {name}: {df.shape}")
        else:
            print(f"  Warning: {name}.txt not found")

    return result


def load_r_output(output_dir: Path) -> dict:
    """Load R output files (supports both H5AD and TXT formats)."""
    
    # Check if it's an H5AD file path
    if output_dir.suffix == '.h5ad' and output_dir.exists():
        return load_r_output_h5ad(output_dir)
    
    # Check for output.h5ad in directory
    h5ad_file = output_dir / "output.h5ad"
    if h5ad_file.exists():
        return load_r_output_h5ad(h5ad_file)
    
    # Fall back to TXT files
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

        # Check shape
        if py_arr.shape != r_arr.shape:
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Shape mismatch: Python {py_arr.shape} vs R {r_arr.shape}'
            }
            continue

        # Check row names (proteins)
        py_rows = set(py_arr.index)
        r_rows = set(r_arr.index)
        if py_rows != r_rows:
            missing_in_py = len(r_rows - py_rows)
            missing_in_r = len(py_rows - r_rows)
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Row mismatch. Missing in Py: {missing_in_py}, Missing in R: {missing_in_r}'
            }
            continue

        # Check column names
        py_cols = set(py_arr.columns)
        r_cols = set(r_arr.columns)
        if py_cols != r_cols:
            missing_in_py = len(r_cols - py_cols)
            missing_in_r = len(py_cols - r_cols)
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Column mismatch. Missing in Py: {missing_in_py}, Missing in R: {missing_in_r}'
            }
            continue

        # Align by row and column names
        py_aligned = py_arr.loc[r_arr.index, r_arr.columns]

        # Calculate difference
        diff = np.abs(py_aligned.values - r_arr.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        
        # Calculate correlation
        py_flat = py_aligned.values.flatten()
        r_flat = r_arr.values.flatten()
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
                'message': f'Max diff: {max_diff:.2e}, Corr: {corr:.10f}'
            }
        else:
            comparison[name] = {
                'status': 'FAIL',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'correlation': corr,
                'message': f'Max diff: {max_diff:.2e}, Corr: {corr:.10f}'
            }

    return comparison


# =============================================================================
# Main Test
# =============================================================================

def main(input_file=None, reference=None, cell_type_col=None, single_cell=False, save_output=False):
    """
    Run scRNA-seq inference validation.

    Parameters
    ----------
    input_file : str, optional
        Path to input H5AD file. If None, uses default test file.
    reference : str, optional
        Path to reference output (H5AD file or folder with TXT files).
        If None, uses default reference directory.
    cell_type_col : str, optional
        Column name for cell type annotations. If None, uses default.
    single_cell : bool, default=False
        If True, run single-cell resolution. If False, run cell-type resolution.
    save_output : bool, default=False
        If True, save results to files.
    """
    resolution = "Single-Cell" if single_cell else "Cell-Type"
    
    # Determine input file
    if input_file is not None:
        input_path = Path(input_file)
    else:
        input_path = DEFAULT_INPUT
    
    # Determine reference path
    if reference is not None:
        reference_path = Path(reference)
    else:
        reference_path = DEFAULT_REFERENCE_SC if single_cell else DEFAULT_REFERENCE_CT
    
    # Determine cell type column
    ct_col = cell_type_col if cell_type_col is not None else CELL_TYPE_COL

    print("=" * 70)
    print(f"SecActPy scRNA-seq Validation Test (CPU) - {resolution} Resolution")
    print("=" * 70)

    # Check anndata
    try:
        import anndata as ad
    except ImportError:
        print("ERROR: anndata is required for this test.")
        return False

    # Check files
    print("\n1. Checking files...")
    if not input_path.exists():
        print(f"   ERROR: Input file not found: {input_path}")
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
        print("   To generate R reference, run:")
        print("   ```R")
        print("   library(RidgeR)")
        print("   Seurat_obj <- readRDS('OV_scRNAseq_CD4.rds')")
        if single_cell:
            print("   Seurat_obj <- SecAct.activity.inference.scRNAseq(Seurat_obj, cellType_meta='Annotation', is_singleCell_level=TRUE)")
            print("   RidgeR::write_secact_to_h5ad(Seurat_obj, 'dataset/output/signature/scRNAseq_sc_res/output.h5ad')")
        else:
            print("   Seurat_obj <- SecAct.activity.inference.scRNAseq(Seurat_obj, cellType_meta='Annotation')")
            print("   RidgeR::write_secact_to_h5ad(Seurat_obj, 'dataset/output/signature/scRNAseq_ct_res/output.h5ad')")
        print("   ```")
        validate = False

    # Load data
    print("\n2. Loading AnnData...")
    adata = ad.read_h5ad(input_path)
    print(f"   Shape: {adata.shape} (cells × genes)")
    print(f"   Cell types: {adata.obs[ct_col].nunique()}")
    print(f"   Cells: {adata.n_obs}")
    
    # Check data characteristics
    if hasattr(adata.X, 'toarray'):
        sample_data = adata.X[:100, :100].toarray()
    else:
        sample_data = adata.X[:100, :100]
    
    nonzero = sample_data[sample_data != 0]
    is_integer = len(nonzero) > 0 and np.allclose(nonzero, np.round(nonzero))
    max_val = np.max(sample_data)
    has_raw = adata.raw is not None
    is_likely_normalized = max_val < 20 and not is_integer
    
    print(f"   Data looks like counts: {is_integer} (max value: {max_val:.2f})")
    print(f"   adata.raw available: {has_raw}")
    
    if is_likely_normalized and not has_raw:
        print("\n   ⚠️  WARNING: Data appears to be log-normalized, not raw counts!")
        print("      Python will use MEAN aggregation (not SUM) for pseudo-bulk.")
        print("      Results may differ from R if R uses raw counts.")
        print("\n      For exact R compatibility, export raw counts from Seurat:")
        print("      ```R")
        print("      library(SeuratDisk)")
        print("      # Make sure raw counts are in @assays$RNA@counts")
        print("      SaveH5Seurat(seurat_obj, 'data.h5seurat')")
        print("      Convert('data.h5seurat', dest='h5ad')")
        print("      # Or manually save counts:")
        print("      writeMM(seurat_obj@assays$RNA@counts, 'counts.mtx')")
        print("      ```")

    # Run inference
    print(f"\n3. Running SecActPy inference (CPU, {resolution})...")
    start_time = time.time()

    try:
        py_result = secact_activity_inference_scrnaseq(
            adata,
            cell_type_col=ct_col,
            is_single_cell_level=single_cell,
            sig_matrix="secact",
            is_group_sig=True,
            is_group_cor=GROUP_COR,
            lambda_=LAMBDA,
            n_rand=NRAND,
            seed=SEED,
            backend="numpy",
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
                corr = result.get('correlation', 'N/A')

                if status == 'PASS':
                    print(f"  {name:8s}: ✓ PASS - {message}")
                else:
                    print(f"  {name:8s}: ✗ {status} - {message}")
                    all_passed = False

            print("\n" + "=" * 70)
            if all_passed:
                print("ALL TESTS PASSED! ✓")
                print(f"SecActPy scRNAseq ({resolution}) produces identical results to RidgeR.")
            else:
                print("SOME TESTS FAILED! ✗")
                print("Check the detailed output above for discrepancies.")
                
                # Show detailed diagnostics
                print("\n--- Diagnostic Info ---")
                
                # Compare first few values
                if 'beta' in py_result and 'beta' in r_result:
                    py_beta = py_result['beta']
                    r_beta = r_result['beta']
                    
                    # Check if indices match
                    py_proteins = list(py_beta.index)
                    r_proteins = list(r_beta.index)
                    py_samples = list(py_beta.columns)
                    r_samples = list(r_beta.columns)
                    
                    print(f"  Python proteins (first 5): {py_proteins[:5]}")
                    print(f"  R proteins (first 5):      {r_proteins[:5]}")
                    print(f"  Python samples (first 5):  {py_samples[:5]}")
                    print(f"  R samples (first 5):       {r_samples[:5]}")
                    
                    # Find max diff location
                    py_aligned = py_beta.loc[r_beta.index, r_beta.columns]
                    diff = np.abs(py_aligned.values - r_beta.values)
                    max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
                    protein_name = r_beta.index[max_idx[0]]
                    sample_name = r_beta.columns[max_idx[1]]
                    print(f"\n  Max diff at: protein='{protein_name}', sample='{sample_name}'")
                    print(f"    Python value: {py_aligned.loc[protein_name, sample_name]:.10f}")
                    print(f"    R value:      {r_beta.loc[protein_name, sample_name]:.10f}")
                
                # Save output for CLI comparison
                print("\n--- Saving Python output for CLI comparison ---")
                try:
                    from secactpy.io import save_results_to_h5ad
                    suffix = "sc_res" if single_cell else "ct_res"
                    py_output_path = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_py_output.h5ad"
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
    print(f"\n{step_num}. Sample output (first 5 proteins, first 5 samples):")
    print(py_result['zscore'].iloc[:5, :5])

    # Activity statistics by cell type
    step_num += 1
    print(f"\n{step_num}. Activity statistics by cell type:")

    if single_cell:
        cell_types = adata.obs[ct_col].values
        cell_names = list(adata.obs_names)

        for ct in sorted(set(cell_types)):
            mask = cell_types == ct
            ct_cells = [c for c, m in zip(cell_names, mask) if m]
            ct_data = py_result['zscore'][ct_cells]
            mean_activity = ct_data.mean(axis=1).sort_values(ascending=False)

            print(f"\n   {ct} ({len(ct_cells)} cells):")
            print(f"     Top 3 active: {', '.join(mean_activity.head(3).index)}")
            print(f"     Z-score range: [{mean_activity.min():.2f}, {mean_activity.max():.2f}]")
    else:
        for col in py_result['zscore'].columns:
            top_proteins = py_result['zscore'][col].sort_values(ascending=False).head(3)
            print(f"\n   {col}:")
            print(f"     Top 3 active: {', '.join(top_proteins.index)}")

    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            from secactpy.io import save_results_to_h5ad
            
            suffix = "sc_res" if single_cell else "ct_res"
            output_h5ad = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_cpu_output.h5ad"
            save_results_to_h5ad(py_result, output_h5ad, verbose=True)
            print(f"   Saved H5AD to: {output_h5ad}")

        except Exception as e:
            print(f"   Could not save: {e}")

    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy scRNA-seq Inference (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cell-type resolution (pseudo-bulk) with defaults
  python tests/test_scrnaseq_cpu.py

  # Single-cell resolution
  python tests/test_scrnaseq_cpu.py --single-cell

  # Custom input and reference
  python tests/test_scrnaseq_cpu.py --input data.h5ad --reference ref.h5ad
  python tests/test_scrnaseq_cpu.py -i data.h5ad -r ref_folder/

  # Custom cell type column
  python tests/test_scrnaseq_cpu.py --input data.h5ad --cell-type-col CellType

  # Save results
  python tests/test_scrnaseq_cpu.py --save
  python tests/test_scrnaseq_cpu.py --single-cell --save
        """
    )
    parser.add_argument(
        '--input', '-i',
        dest='input_file',
        default=None,
        help='Path to input H5AD file (default: test dataset)'
    )
    parser.add_argument(
        '--reference', '-r',
        dest='reference',
        default=None,
        help='Path to R reference output (H5AD file or folder with TXT files)'
    )
    parser.add_argument(
        '--cell-type-col',
        dest='cell_type_col',
        default=None,
        help='Column name for cell type annotations (default: Annotation)'
    )
    parser.add_argument(
        '--single-cell',
        action='store_true',
        help='Run single-cell resolution (default: cell-type resolution)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(
        input_file=args.input_file,
        reference=args.reference,
        cell_type_col=args.cell_type_col,
        single_cell=args.single_cell,
        save_output=args.save
    )
    sys.exit(0 if success else 1)
