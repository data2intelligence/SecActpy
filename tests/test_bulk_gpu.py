#!/usr/bin/env python3
"""
Test Script: GPU-Accelerated Bulk RNA-seq Inference

Compares CPU vs GPU computation for bulk RNA-seq activity inference.

Requirements:
    - CuPy installed (pip install cupy-cuda11x or cupy-cuda12x)
    - NVIDIA GPU with CUDA support

Usage:
    python tests/test_bulk_gpu.py
    python tests/test_bulk_gpu.py --input data.txt
    python tests/test_bulk_gpu.py --save
    python tests/test_bulk_gpu.py --resample 1000      # Test with 1000 resampled samples
    python tests/test_bulk_gpu.py --resample 1000 --gpu-only  # Skip CPU, benchmark GPU only

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

from secactpy import secact_activity_inference, load_expression_data
from secactpy.ridge import CUPY_AVAILABLE


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
INPUT_FILE = DATA_DIR / "input" / "Ly86-Fc_vs_Vehicle_logFC.txt"

# Parameters
LAMBDA = 5e5
NRAND = 1000
SEED = 0
GROUP_COR = 0.9


# =============================================================================
# Resampling Function
# =============================================================================

def resample_expression(Y: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Resample expression data to create more samples for benchmarking.
    """
    np.random.seed(seed)

    n_genes = Y.shape[0]
    n_original = Y.shape[1]

    # Resample columns with replacement
    sample_indices = np.random.choice(n_original, size=n_samples, replace=True)

    # Create resampled data
    resampled = Y.iloc[:, sample_indices].copy()

    # Add small random noise to create variation
    data_std = np.std(Y.values)
    noise = np.random.normal(0, data_std * 0.01, size=(n_genes, n_samples))
    resampled_values = resampled.values + noise

    # Create new column names
    new_columns = [f"Sample_{i+1}" for i in range(n_samples)]

    return pd.DataFrame(resampled_values, index=Y.index, columns=new_columns)


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

def main(input_file=None, save_output=False, resample=None, gpu_only=False):
    """
    Run GPU vs CPU comparison for bulk RNA-seq.

    Parameters
    ----------
    input_file : str, optional
        Path to input expression file. If None, uses default test file.
    save_output : bool, default=False
        If True, save results to files.
    resample : int, optional
        If provided, resample to this many samples for benchmarking.
    gpu_only : bool, default=False
        If True, skip CPU test (for benchmarking GPU only).
    """
    print("=" * 70)
    print("SecActPy GPU vs CPU Comparison: Bulk RNA-seq")
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

    # Determine input file
    input_path = Path(input_file) if input_file is not None else INPUT_FILE

    # Check input file
    print("\n2. Checking input file...")
    if not input_path.exists():
        print(f"   ERROR: Input file not found: {input_path}")
        return False
    print(f"   Input: {input_path}")

    if resample is not None:
        print(f"   Resampling: {resample} samples")

    if gpu_only:
        print(f"   Mode: GPU-only benchmark (skipping CPU)")

    # Load data
    print("\n3. Loading input data...")
    Y = load_expression_data(input_path)
    print(f"   Original expression data: {Y.shape} (genes × samples)")

    # Resample if requested
    if resample is not None:
        print(f"\n   Resampling to {resample} samples...")
        Y = resample_expression(Y, n_samples=resample, seed=42)
        print(f"   Resampled expression data: {Y.shape} (genes × samples)")
        print(f"   Memory: {Y.values.nbytes / 1e6:.1f} MB")

    cpu_result = None
    cpu_time = None

    # Run CPU inference (unless gpu_only)
    if not gpu_only:
        print("\n4. Running CPU inference...")
        cpu_start = time.time()

        cpu_result = secact_activity_inference(
            input_profile=Y,
            is_differential=True,
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
        step_num = 5
    else:
        print("\n4. Skipping CPU inference (--gpu-only)")
        step_num = 5

    # Run GPU inference
    print(f"\n{step_num}. Running GPU inference...")
    gpu_start = time.time()

    gpu_result = secact_activity_inference(
        input_profile=Y,
        is_differential=True,
        sig_matrix="secact",
        is_group_sig=True,
        is_group_cor=GROUP_COR,
        lambda_=LAMBDA,
        n_rand=NRAND,
        seed=SEED,
        backend="cupy",  # GPU
        use_cache=True,
        verbose=True
    )

    gpu_time = time.time() - gpu_start
    print(f"   GPU time: {gpu_time:.2f} seconds")
    print(f"   Result shape: {gpu_result['beta'].shape}")

    all_passed = True

    # Compare results (if CPU was run)
    if cpu_result is not None:
        step_num += 1
        print(f"\n{step_num}. Comparing CPU vs GPU results...")
        comparison = compare_results(cpu_result, gpu_result)

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

    print("\n" + "-" * 70)
    print("PERFORMANCE")
    print("-" * 70)
    n_samples = Y.shape[1]
    if cpu_time is not None:
        print(f"  CPU time: {cpu_time:.2f} seconds")
        print(f"  CPU throughput: {n_samples / cpu_time:.0f} samples/sec")
    print(f"  GPU time: {gpu_time:.2f} seconds")
    print(f"  GPU throughput: {n_samples / gpu_time:.0f} samples/sec")
    if cpu_time is not None:
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  Speedup:  {speedup:.2f}x")
    print(f"  Samples:  {n_samples}")

    print("\n" + "=" * 70)
    if cpu_result is not None:
        if all_passed:
            print("ALL TESTS PASSED! ✓")
            print("GPU produces identical results to CPU.")
        else:
            print("SOME TESTS FAILED! ✗")
            print("GPU results differ from CPU.")
    else:
        print("GPU BENCHMARK COMPLETE ✓")
    print("=" * 70)

    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            from secactpy.io import save_results

            output_h5 = PACKAGE_ROOT / "dataset" / "output" / "bulk_gpu_activity.h5"
            results_to_save = {
                'beta': gpu_result['beta'].values,
                'se': gpu_result['se'].values,
                'zscore': gpu_result['zscore'].values,
                'pvalue': gpu_result['pvalue'].values,
                'feature_names': list(gpu_result['beta'].index),
                'sample_names': list(gpu_result['beta'].columns),
            }
            save_results(results_to_save, output_h5)
            print(f"   Saved to: {output_h5}")
        except Exception as e:
            print(f"   Could not save: {e}")

    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy GPU vs CPU Comparison: Bulk RNA-seq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default test file (compare CPU vs GPU)
  python tests/test_bulk_gpu.py

  # Run with custom input file
  python tests/test_bulk_gpu.py --input data.txt
  python tests/test_bulk_gpu.py -i data.csv

  # Run with resampled data (1000 samples)
  python tests/test_bulk_gpu.py --resample 1000

  # Benchmark GPU only (skip CPU comparison)
  python tests/test_bulk_gpu.py --resample 10000 --gpu-only

  # Save results to HDF5
  python tests/test_bulk_gpu.py --save
        """
    )
    parser.add_argument(
        '--input', '-i',
        dest='input_file',
        default=None,
        help='Path to expression data file (default: test dataset)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save GPU results to HDF5 file'
    )
    parser.add_argument(
        '--resample',
        type=int,
        default=None,
        metavar='N',
        help='Resample to N samples for benchmarking (e.g., --resample 1000)'
    )
    parser.add_argument(
        '--gpu-only',
        action='store_true',
        help='Skip CPU comparison (benchmark GPU only)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(
        input_file=args.input_file,
        save_output=args.save,
        resample=args.resample,
        gpu_only=args.gpu_only
    )
    sys.exit(0 if success else 1)
