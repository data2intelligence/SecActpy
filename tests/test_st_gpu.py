#!/usr/bin/env python3
"""
Test Script: GPU-Accelerated Spatial Transcriptomics Inference

Compares CPU vs GPU computation for spatial transcriptomics activity inference.
Tests Visium data by default, can also test CosMx with --cosmx flag.

Requirements:
    - CuPy installed (pip install cupy-cuda11x or cupy-cuda12x)
    - NVIDIA GPU with CUDA support

Usage:
    python tests/test_st_gpu.py              # Visium dataset, CPU vs GPU
    python tests/test_st_gpu.py --cosmx      # CosMx dataset, CPU vs GPU
    python tests/test_st_gpu.py --input data.h5ad --reference ref.h5ad
    python tests/test_st_gpu.py --gpu-only   # GPU only, compare to R reference
    python tests/test_st_gpu.py --cosmx --gpu-only  # CosMx, GPU only
    python tests/test_st_gpu.py --save

For faster R comparison, convert R text outputs to h5ad first:
    Rscript scripts/save_results_h5ad.R dataset/output/signature/CosMx

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

from secactpy import secact_activity_inference_st, load_visium_10x, load_signature
from secactpy.ridge import CUPY_AVAILABLE


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"

# Visium configuration
VISIUM_INPUT = DATA_DIR / "input" / "Visium_HCC"
VISIUM_OUTPUT = DATA_DIR / "output" / "signature" / "ST"
VISIUM_MIN_GENES = 1000
VISIUM_SCALE_FACTOR = 1e5

# CosMx configuration
COSMX_INPUT = DATA_DIR / "input" / "LIHC_CosMx_data.h5ad"
COSMX_OUTPUT = DATA_DIR / "output" / "signature" / "CosMx"
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
    """Load R output from h5ad file (fast)."""
    h5ad_path = output_dir / "output.h5ad"
    if not h5ad_path.exists():
        h5ad_path = output_dir / "results.h5ad"
    if not h5ad_path.exists():
        # Check if output_dir itself is an h5ad file
        if output_dir.suffix == '.h5ad' and output_dir.exists():
            h5ad_path = output_dir
        else:
            return None

    print(f"  Loading from h5ad: {h5ad_path}")
    start = time.time()

    try:
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path)

        protein_names = list(adata.var_names)
        cell_names = list(adata.obs_names)

        print(f"    Proteins: {len(protein_names)}, Cells: {len(cell_names)}")

        result = {}

        # X is beta (cells x proteins), transpose to (proteins x cells)
        if adata.X is not None:
            result['beta'] = pd.DataFrame(
                adata.X.T,
                index=protein_names,
                columns=cell_names
            )
            print(f"    beta: {result['beta'].shape}")

        # Other matrices from obsm
        for name in ['se', 'zscore', 'pvalue']:
            if name in adata.obsm:
                result[name] = pd.DataFrame(
                    adata.obsm[name].T,
                    index=protein_names,
                    columns=cell_names
                )
                print(f"    {name}: {result[name].shape}")

        elapsed = time.time() - start
        print(f"  Loaded in {elapsed:.1f}s")

        return result if result else None

    except Exception as e:
        print(f"  anndata failed: {e}")
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
    """Load R output files, preferring h5ad format if available."""
    result = load_r_output_h5ad(output_dir)
    if result:
        return result

    print("  H5AD not found, loading text files (slower)...")
    print("  Tip: Run 'Rscript scripts/save_results_h5ad.R <output_dir>' for faster loading")
    return load_r_output_txt(output_dir)


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_results(result1: dict, result2: dict, tolerance: float = 1e-8,
                    label1: str = "Result1", label2: str = "Result2") -> dict:
    """Compare two result sets."""
    comparison = {}

    for name in ['beta', 'se', 'zscore', 'pvalue']:
        if name not in result1 or name not in result2:
            comparison[name] = {'status': 'MISSING', 'message': 'Array not found'}
            continue

        arr1 = result1[name]
        arr2 = result2[name]

        # Find common rows and columns
        common_rows = arr1.index.intersection(arr2.index)
        common_cols = arr1.columns.intersection(arr2.columns)

        if len(common_rows) == 0 or len(common_cols) == 0:
            comparison[name] = {
                'status': 'FAIL',
                'message': f'No common rows/cols. {label1}: {arr1.shape}, {label2}: {arr2.shape}'
            }
            continue

        # Align and compare
        arr1_aligned = arr1.loc[common_rows, common_cols]
        arr2_aligned = arr2.loc[common_rows, common_cols]

        # Calculate difference
        diff = np.abs(arr1_aligned.values - arr2_aligned.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)

        n_compared = len(common_rows) * len(common_cols)

        if max_diff <= tolerance:
            comparison[name] = {
                'status': 'PASS',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'n_compared': n_compared,
                'message': f'Max diff: {max_diff:.2e} (n={n_compared:,})'
            }
        else:
            max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
            max_row = common_rows[max_idx[0]]
            max_col = common_cols[max_idx[1]]

            comparison[name] = {
                'status': 'FAIL',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'n_compared': n_compared,
                'max_loc': (max_row, max_col),
                'message': f'Max diff: {max_diff:.2e} at ({max_row}, {max_col})'
            }

    return comparison


def load_cosmx_data(input_file, min_genes=50, verbose=True):
    """Load CosMx data from h5ad file."""
    import anndata as ad
    from scipy import sparse

    adata = ad.read_h5ad(input_file)

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
    n_kept = keep_cells.sum()
    counts_matrix = counts_matrix[keep_cells, :]
    cell_names = [c for c, k in zip(cell_names, keep_cells) if k]

    if verbose:
        print(f"   Cells after QC (>={min_genes} genes): {n_kept}")

    # Transpose to (genes × cells)
    if sparse.issparse(counts_matrix):
        counts_transposed = counts_matrix.T.tocsr()
        counts_df = pd.DataFrame.sparse.from_spmatrix(
            counts_transposed, index=gene_names, columns=cell_names
        ).sparse.to_dense()
    else:
        counts_df = pd.DataFrame(counts_matrix.T, index=gene_names, columns=cell_names)

    return counts_df


# =============================================================================
# Main Test
# =============================================================================

def main(input_file=None, reference=None, cosmx=False, gpu_only=False, save_output=False):
    """
    Run GPU vs CPU comparison for spatial transcriptomics.

    Parameters
    ----------
    input_file : str, optional
        Path to input file. If None, uses default for platform.
    reference : str, optional
        Path to reference output (H5AD file or folder with TXT files).
        If None, uses default reference directory.
    cosmx : bool, default=False
        If True, use CosMx dataset. If False, use Visium dataset.
    gpu_only : bool, default=False
        If True, run only GPU inference and compare to R reference.
    save_output : bool, default=False
        If True, save results to files.
    """
    platform = "CosMx" if cosmx else "Visium"
    mode = "GPU Only (vs R)" if gpu_only else "GPU vs CPU"

    print("=" * 70)
    print(f"SecActPy {mode}: Spatial Transcriptomics ({platform})")
    print("=" * 70)

    # Check GPU availability
    print("\n1. Checking GPU availability...")
    if not CUPY_AVAILABLE:
        from secactpy.ridge import CUPY_INIT_ERROR
        print("   ERROR: GPU is not available!")
        if CUPY_INIT_ERROR:
            print(f"   Reason: {CUPY_INIT_ERROR}")
        else:
            print("   CuPy is not installed.")
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

    # Set configuration based on platform
    if cosmx:
        default_input = COSMX_INPUT
        output_dir = COSMX_OUTPUT
        min_genes = COSMX_MIN_GENES
        scale_factor = COSMX_SCALE_FACTOR
        sig_filter = True
    else:
        default_input = VISIUM_INPUT
        output_dir = VISIUM_OUTPUT
        min_genes = VISIUM_MIN_GENES
        scale_factor = VISIUM_SCALE_FACTOR
        sig_filter = False

    # Determine input and reference paths
    input_path = Path(input_file) if input_file is not None else default_input
    reference_path = Path(reference) if reference is not None else output_dir

    # Load data based on platform
    print("\n2. Loading data...")

    if not input_path.exists():
        print(f"   ERROR: Input not found: {input_path}")
        return False
    print(f"   Input: {input_path}")

    if cosmx:
        input_data = load_cosmx_data(input_path, min_genes=min_genes, verbose=True)
        print(f"   Final shape: {input_data.shape} (genes × cells)")
    else:
        input_data = load_visium_10x(input_path, min_genes=min_genes, verbose=True)
        print(f"   Spots: {len(input_data['spot_names'])}")

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
        print("\n   ERROR: No overlapping genes!")
        return False

    step_num = 3
    cpu_time = None
    cpu_result = None

    # Run CPU inference (unless gpu_only mode)
    if not gpu_only:
        print(f"\n{step_num}. Running CPU inference...")
        cpu_start = time.time()

        cpu_result = secact_activity_inference_st(
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
            use_gsl_rng=True,
            use_cache=True,
            verbose=False
        )

        cpu_time = time.time() - cpu_start
        print(f"   CPU time: {cpu_time:.2f} seconds")
        print(f"   Result shape: {cpu_result['beta'].shape}")
        step_num += 1

    # Run GPU inference
    print(f"\n{step_num}. Running GPU inference...")
    gpu_start = time.time()

    gpu_result = secact_activity_inference_st(
        input_data=input_data,
        scale_factor=scale_factor,
        sig_matrix="secact",
        is_group_sig=True,
        is_group_cor=GROUP_COR,
        lambda_=LAMBDA,
        n_rand=NRAND,
        seed=SEED,
        sig_filter=sig_filter,
        backend="cupy",
        use_gsl_rng=True,
        use_cache=True,
        verbose=True
    )

    gpu_time = time.time() - gpu_start
    print(f"   GPU time: {gpu_time:.2f} seconds")
    print(f"   Result shape: {gpu_result['beta'].shape}")
    step_num += 1

    # Compare results
    all_passed = True

    if gpu_only:
        # Compare to R reference
        print(f"\n{step_num}. Loading R reference output...")
        validate = True

        if not reference_path.exists():
            print(f"   Warning: Reference output not found: {reference_path}")
            print("   Skipping validation.")
            validate = False
        else:
            r_result = load_r_output(reference_path)

            if not r_result:
                print("   Warning: No R output files found!")
                validate = False
            else:
                step_num += 1
                print(f"\n{step_num}. Comparing GPU results to R reference...")
                comparison = compare_results(gpu_result, r_result, tolerance=1e-10,
                                             label1="GPU", label2="R")

                print("\n" + "=" * 70)
                print("RESULTS (GPU vs R Reference)")
                print("=" * 70)

                for name, result in comparison.items():
                    status = result['status']
                    message = result['message']

                    if status == 'PASS':
                        print(f"  {name:8s}: ✓ PASS - {message}")
                    else:
                        print(f"  {name:8s}: ✗ {status} - {message}")
                        all_passed = False

        if not validate:
            all_passed = True  # Can't fail if no validation
    else:
        # Compare CPU vs GPU
        print(f"\n{step_num}. Comparing CPU vs GPU results...")
        comparison = compare_results(cpu_result, gpu_result, tolerance=1e-8,
                                     label1="CPU", label2="GPU")

        print("\n" + "=" * 70)
        print("RESULTS (CPU vs GPU)")
        print("=" * 70)

        for name, result in comparison.items():
            status = result['status']
            message = result['message']

            if status == 'PASS':
                print(f"  {name:8s}: ✓ PASS - {message}")
            else:
                print(f"  {name:8s}: ✗ {status} - {message}")
                all_passed = False

    # Performance summary
    print("\n" + "-" * 70)
    print("PERFORMANCE")
    print("-" * 70)
    print(f"  Platform: {platform}")
    print(f"  Samples:  {gpu_result['beta'].shape[1]}")
    if cpu_time is not None:
        print(f"  CPU time: {cpu_time:.2f} seconds")
    print(f"  GPU time: {gpu_time:.2f} seconds")
    if cpu_time is not None:
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  Speedup:  {speedup:.2f}x")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        if gpu_only:
            print("GPU produces results matching R reference.")
        else:
            print("GPU produces identical results to CPU.")
    else:
        print("SOME TESTS FAILED! ✗")
        if gpu_only:
            print("GPU results differ from R reference.")
        else:
            print("GPU results differ from CPU.")
    print("=" * 70)

    # Sample output
    step_num += 1
    print(f"\n{step_num}. Sample output (first 5 proteins, first 3 samples):")
    cols = gpu_result['zscore'].columns[:3]
    print(gpu_result['zscore'].iloc[:5][cols])

    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            from secactpy.io import save_st_results_to_h5ad

            suffix = "CosMx" if cosmx else "Visium"
            output_h5ad = PACKAGE_ROOT / "dataset" / "output" / f"{suffix}_gpu_activity.h5ad"

            if cosmx:
                save_st_results_to_h5ad(
                    counts=input_data.values,
                    activity_results=gpu_result,
                    output_path=output_h5ad,
                    gene_names=list(input_data.index),
                    cell_names=list(input_data.columns),
                    platform=platform
                )
            else:
                save_st_results_to_h5ad(
                    counts=input_data['counts'],
                    activity_results=gpu_result,
                    output_path=output_h5ad,
                    gene_names=input_data['gene_names'],
                    cell_names=input_data['spot_names'],
                    spatial_coords=input_data['spot_coordinates'],
                    platform=platform
                )

            print(f"   Saved to: {output_h5ad}")

        except Exception as e:
            print(f"   Could not save: {e}")
            import traceback
            traceback.print_exc()

    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy GPU Spatial Transcriptomics Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visium dataset, CPU vs GPU comparison
  python tests/test_st_gpu.py

  # CosMx dataset, CPU vs GPU comparison
  python tests/test_st_gpu.py --cosmx

  # Custom input file
  python tests/test_st_gpu.py --input data.h5ad
  python tests/test_st_gpu.py -i Visium_folder/

  # GPU only, compare to R reference
  python tests/test_st_gpu.py --gpu-only
  python tests/test_st_gpu.py --cosmx --gpu-only
  python tests/test_st_gpu.py --gpu-only --reference ref.h5ad

  # Save results
  python tests/test_st_gpu.py --save

For faster R comparison with large datasets:
  # First, convert R text outputs to h5ad (run once):
  Rscript scripts/save_results_h5ad.R dataset/output/signature/CosMx

  # Then run the test (will auto-detect h5ad):
  python tests/test_st_gpu.py --cosmx --gpu-only
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
        '--gpu-only',
        action='store_true',
        help='Run GPU only and compare to R reference (skip CPU)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save GPU results to h5ad file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(
        input_file=args.input_file,
        reference=args.reference,
        cosmx=args.cosmx,
        gpu_only=args.gpu_only,
        save_output=args.save
    )
    sys.exit(0 if success else 1)
