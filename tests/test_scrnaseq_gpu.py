#!/usr/bin/env python3
"""
Test Script: GPU-Accelerated scRNA-seq Inference

Compares CPU vs GPU computation for scRNA-seq activity inference.
Tests both cell-type resolution (pseudo-bulk) and single-cell resolution.

Requirements:
    - CuPy installed (pip install cupy-cuda11x or cupy-cuda12x)
    - NVIDIA GPU with CUDA support
    - anndata installed

Usage:
    python tests/test_scrnaseq_gpu.py
    python tests/test_scrnaseq_gpu.py --input data.h5ad --cell-type-col CellType
    python tests/test_scrnaseq_gpu.py --single-cell
    python tests/test_scrnaseq_gpu.py --save

Expected output:
    GPU and CPU results should match within numerical tolerance.
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
from secactpy.ridge import CUPY_AVAILABLE


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
INPUT_FILE = DATA_DIR / "input" / "OV_scRNAseq_data.h5ad"

# Parameters
CELL_TYPE_COL = "Annotation"
LAMBDA = 5e5
NRAND = 1000
SEED = 0
GROUP_COR = 0.9


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_results(cpu_result: dict, gpu_result: dict, tolerance: float = 1e-8) -> dict:
    """Compare CPU and GPU results."""
    comparison = {}

    for name in ['beta', 'se', 'zscore', 'pvalue']:
        if name not in cpu_result or name not in gpu_result:
            comparison[name] = {'status': 'MISSING', 'message': 'Array not found'}
            continue

        cpu_arr = cpu_result[name]
        gpu_arr = gpu_result[name]

        # Check shape
        if cpu_arr.shape != gpu_arr.shape:
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Shape mismatch: CPU {cpu_arr.shape} vs GPU {gpu_arr.shape}'
            }
            continue

        # Align by index/columns
        gpu_aligned = gpu_arr.loc[cpu_arr.index, cpu_arr.columns]

        # Calculate difference
        diff = np.abs(cpu_arr.values - gpu_aligned.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)

        if max_diff <= tolerance:
            comparison[name] = {
                'status': 'PASS',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'message': f'Max diff: {max_diff:.2e}'
            }
        else:
            comparison[name] = {
                'status': 'FAIL',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'message': f'Max diff: {max_diff:.2e} (tolerance: {tolerance:.2e})'
            }

    return comparison


# =============================================================================
# Main Test
# =============================================================================

def main(input_file=None, cell_type_col=None, single_cell=False, save_output=False):
    """
    Run GPU vs CPU comparison for scRNA-seq.

    Parameters
    ----------
    input_file : str, optional
        Path to input H5AD file. If None, uses default test file.
    cell_type_col : str, optional
        Column name for cell type annotations. If None, uses default.
    single_cell : bool, default=False
        If True, run single-cell resolution. If False, run cell-type resolution.
    save_output : bool, default=False
        If True, save results to files.
    """
    resolution = "Single-Cell" if single_cell else "Cell-Type"
    
    # Determine input file and cell type column
    input_path = Path(input_file) if input_file is not None else INPUT_FILE
    ct_col = cell_type_col if cell_type_col is not None else CELL_TYPE_COL

    print("=" * 70)
    print(f"SecActPy GPU vs CPU Comparison: scRNA-seq ({resolution} Resolution)")
    print("=" * 70)

    # Check GPU availability
    print("\n1. Checking GPU availability...")
    if not CUPY_AVAILABLE:
        print("   ERROR: CuPy is not available!")
        print("   Install with: pip install cupy-cuda11x  # or cupy-cuda12x")
        print("   Skipping GPU test.")
        return False

    try:
        import cupy as cp
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        print(f"   ✓ CuPy available")
        print(f"   GPU: {gpu_info['name'].decode()}")
        print(f"   Memory: {gpu_info['totalGlobalMem'] / 1e9:.1f} GB")
    except Exception as e:
        print(f"   ERROR: Could not initialize GPU: {e}")
        return False

    # Check anndata
    try:
        import anndata as ad
    except ImportError:
        print("   ERROR: anndata is required for this test.")
        print("   Install with: pip install anndata")
        return False

    # Check input file
    print("\n2. Checking input file...")
    if not input_path.exists():
        print(f"   ERROR: Input file not found: {input_path}")
        return False
    print(f"   Input: {input_path}")

    # Load data
    print("\n3. Loading AnnData...")
    adata = ad.read_h5ad(input_path)
    print(f"   Shape: {adata.shape} (cells × genes)")
    print(f"   Cell types: {adata.obs[ct_col].nunique()}")

    # Run CPU inference
    print("\n4. Running CPU inference...")
    cpu_start = time.time()

    cpu_result = secact_activity_inference_scrnaseq(
        adata,
        cell_type_col=ct_col,
        is_single_cell_level=single_cell,
        sig_matrix="secact",
        is_group_sig=True,
        is_group_cor=GROUP_COR,
        lambda_=LAMBDA,
        n_rand=NRAND,
        seed=SEED,
        backend="numpy",  # CPU
        use_cache=True,
        verbose=False
    )

    cpu_time = time.time() - cpu_start
    print(f"   CPU time: {cpu_time:.2f} seconds")
    print(f"   Result shape: {cpu_result['beta'].shape}")

    # Run GPU inference
    print("\n5. Running GPU inference...")
    gpu_start = time.time()

    gpu_result = secact_activity_inference_scrnaseq(
        adata,
        cell_type_col=ct_col,
        is_single_cell_level=single_cell,
        sig_matrix="secact",
        is_group_sig=True,
        is_group_cor=GROUP_COR,
        lambda_=LAMBDA,
        n_rand=NRAND,
        seed=SEED,
        backend="cupy",  # GPU
        use_cache=True,
        verbose=False
    )

    gpu_time = time.time() - gpu_start
    print(f"   GPU time: {gpu_time:.2f} seconds")
    print(f"   Result shape: {gpu_result['beta'].shape}")

    # Compare results
    print("\n6. Comparing CPU vs GPU results...")
    comparison = compare_results(cpu_result, gpu_result)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_passed = True
    for name, result in comparison.items():
        status = result['status']
        message = result['message']

        if status == 'PASS':
            print(f"  {name:8s}: ✓ PASS - {message}")
        else:
            print(f"  {name:8s}: ✗ {status} - {message}")
            all_passed = False

    print("\n" + "-" * 70)
    print("PERFORMANCE")
    print("-" * 70)
    print(f"  Resolution: {resolution}")
    print(f"  Samples:    {cpu_result['beta'].shape[1]}")
    print(f"  CPU time:   {cpu_time:.2f} seconds")
    print(f"  GPU time:   {gpu_time:.2f} seconds")
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    print(f"  Speedup:    {speedup:.2f}x")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("GPU produces identical results to CPU.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("GPU results differ from CPU.")
    print("=" * 70)

    # Save results
    if save_output:
        print("\n7. Saving results...")
        try:
            suffix = "sc_res" if single_cell else "ct_res"

            if single_cell:
                from secactpy.io import add_activity_to_anndata
                adata = add_activity_to_anndata(adata, gpu_result)
                output_h5ad = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_gpu_activity.h5ad"
                adata.write_h5ad(output_h5ad)
                print(f"   Saved h5ad to: {output_h5ad}")
            else:
                from secactpy.io import save_results
                output_h5 = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_gpu_activity.h5"
                results_to_save = {
                    'beta': gpu_result['beta'].values,
                    'se': gpu_result['se'].values,
                    'zscore': gpu_result['zscore'].values,
                    'pvalue': gpu_result['pvalue'].values,
                    'feature_names': list(gpu_result['beta'].index),
                    'sample_names': list(gpu_result['beta'].columns),
                }
                save_results(results_to_save, output_h5)
                print(f"   Saved HDF5 to: {output_h5}")

        except Exception as e:
            print(f"   Could not save: {e}")

    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy GPU vs CPU Comparison: scRNA-seq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default test file (cell-type resolution)
  python tests/test_scrnaseq_gpu.py

  # Run with custom input file
  python tests/test_scrnaseq_gpu.py --input data.h5ad
  python tests/test_scrnaseq_gpu.py -i data.h5ad --cell-type-col CellType

  # Run single-cell resolution
  python tests/test_scrnaseq_gpu.py --single-cell

  # Save results
  python tests/test_scrnaseq_gpu.py --save
        """
    )
    parser.add_argument(
        '--input', '-i',
        dest='input_file',
        default=None,
        help='Path to input H5AD file (default: test dataset)'
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
        help='Save GPU results to file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(
        input_file=args.input_file,
        cell_type_col=args.cell_type_col,
        single_cell=args.single_cell,
        save_output=args.save
    )
    sys.exit(0 if success else 1)
