#!/usr/bin/env python3
"""
Test Script: Core Ridge Regression Validation (No Grouping)

Validates SecActPy's core ridge regression against SecAct's 
SecAct.inference.gsl output (no signature grouping).

R code to generate reference:
    library(SecAct)
    dataPath <- file.path(system.file(package = "SecAct"), "extdata")
    expr.diff <- read.table(paste0(dataPath, "/Ly86-Fc_vs_Vehicle_logFC.txt"))
    res <- SecAct.inference.gsl(expr.diff)
    
    write.table(res$beta, "beta.txt", quote=F)
    write.table(res$se, "se.txt", quote=F)
    write.table(res$zscore, "zscore.txt", quote=F)
    write.table(res$pvalue, "pvalue.txt", quote=F)

Usage:
    python tests/test_ridge.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add package to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy import load_signature
from secactpy.ridge import ridge


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
INPUT_FILE = DATA_DIR / "input" / "Ly86-Fc_vs_Vehicle_logFC.txt"
OUTPUT_DIR = DATA_DIR / "output" / "ridge" / "bulk"

# Parameters matching R's SecAct.inference.gsl.legacy defaults
LAMBDA = 5e5
NRAND = 1000
SEED = 0


# =============================================================================
# Core Ridge Inference (matching SecAct.inference.gsl.legacy)
# =============================================================================

def secact_inference_gsl_legacy(
    Y: pd.DataFrame,
    sig_matrix: str = "secact",
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    verbose: bool = True
) -> dict:
    """
    Ridge regression matching R's SecAct.inference.gsl.legacy.
    
    NO signature grouping - just:
    1. Load signature
    2. Find overlapping genes
    3. Scale both matrices
    4. Run ridge regression
    """
    # Load signature
    X = load_signature(sig_matrix)
    n_sig_genes = X.shape[0]
    
    if verbose:
        print(f"  Loaded signature: {n_sig_genes} genes × {X.shape[1]} proteins")
    
    # Find overlapping genes
    common_genes = Y.index.intersection(X.index)
    if verbose:
        print(f"  Common genes: {len(common_genes)}")
    
    if len(common_genes) < 2:
        raise ValueError(f"Too few overlapping genes: {len(common_genes)}")
    
    # Subset to common genes
    X_aligned = X.loc[common_genes].astype(np.float64)
    Y_aligned = Y.loc[common_genes].astype(np.float64)
    
    # Scale (R's scale() uses ddof=1)
    X_scaled = (X_aligned - X_aligned.mean()) / X_aligned.std(ddof=1)
    Y_scaled = (Y_aligned - Y_aligned.mean()) / Y_aligned.std(ddof=1)
    
    # Replace NaN with 0
    X_scaled = X_scaled.fillna(0)
    Y_scaled = Y_scaled.fillna(0)
    
    if verbose:
        print(f"  Running ridge regression (n_rand={n_rand})...")
    
    # Run ridge (use_cache=True since we run multiple times on same dataset in tests)
    result = ridge(
        X=X_scaled.values,
        Y=Y_scaled.values,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        backend='numpy',
        use_cache=True,
        verbose=False
    )
    
    # Create DataFrames with proper labels
    feature_names = X_scaled.columns.tolist()
    sample_names = Y_scaled.columns.tolist()
    
    return {
        'beta': pd.DataFrame(result['beta'], index=feature_names, columns=sample_names),
        'se': pd.DataFrame(result['se'], index=feature_names, columns=sample_names),
        'zscore': pd.DataFrame(result['zscore'], index=feature_names, columns=sample_names),
        'pvalue': pd.DataFrame(result['pvalue'], index=feature_names, columns=sample_names),
    }


# =============================================================================
# Comparison Functions
# =============================================================================

def load_r_output(output_dir: Path) -> dict:
    """Load R output files."""
    result = {}
    
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        filepath = output_dir / f"{name}.txt"
        if filepath.exists():
            df = pd.read_csv(filepath, sep=r'\s+', index_col=0)
            result[name] = df
            print(f"  Loaded {name}: {df.shape}")
        else:
            print(f"  Warning: {filepath} not found")
    
    return result


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
        
        # Check row names
        py_rows = set(py_arr.index)
        r_rows = set(r_arr.index)
        if py_rows != r_rows:
            missing_in_py = r_rows - py_rows
            missing_in_r = py_rows - r_rows
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Row mismatch. Missing in Py: {len(missing_in_py)}, in R: {len(missing_in_r)}'
            }
            continue
        
        # Align and compare
        py_aligned = py_arr.loc[r_arr.index]
        diff = np.abs(py_aligned.values - r_arr.values)
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
            max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
            row_name = py_aligned.index[max_idx[0]]
            col_name = py_aligned.columns[max_idx[1]]
            py_val = py_aligned.iloc[max_idx[0], max_idx[1]]
            r_val = r_arr.iloc[max_idx[0], max_idx[1]]
            
            comparison[name] = {
                'status': 'FAIL',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'location': f'{row_name}, {col_name}',
                'py_value': py_val,
                'r_value': r_val,
                'message': f'Max diff: {max_diff:.2e} at ({row_name}, {col_name}): Py={py_val:.6f}, R={r_val:.6f}'
            }
    
    return comparison


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 70)
    print("SecActPy Core Ridge Regression Validation")
    print("(Comparing with SecAct.inference.gsl.legacy - no grouping)")
    print("=" * 70)
    
    # Check files
    print("\n1. Checking files...")
    if not INPUT_FILE.exists():
        print(f"   ERROR: Input not found: {INPUT_FILE}")
        return False
    print(f"   Input: {INPUT_FILE}")
    
    if not OUTPUT_DIR.exists():
        print(f"   ERROR: Output dir not found: {OUTPUT_DIR}")
        print(f"   Please create R reference output first.")
        return False
    print(f"   Reference: {OUTPUT_DIR}")
    
    # Load input
    print("\n2. Loading input data...")
    Y = pd.read_csv(INPUT_FILE, sep=r'\s+', index_col=0)
    print(f"   Shape: {Y.shape}")
    print(f"   Columns: {Y.columns.tolist()}")
    print(f"   First 5 genes: {Y.index[:5].tolist()}")
    
    # Run Python inference
    print("\n3. Running SecActPy ridge regression...")
    try:
        py_result = secact_inference_gsl_legacy(
            Y=Y,
            sig_matrix="secact",
            lambda_=LAMBDA,
            n_rand=NRAND,
            seed=SEED,
            verbose=True
        )
        print(f"   Result shape: {py_result['beta'].shape}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load R reference
    print("\n4. Loading R reference output...")
    r_result = load_r_output(OUTPUT_DIR)
    
    if not r_result:
        print("   ERROR: No R output files found!")
        return False
    
    # Compare
    print("\n5. Comparing results...")
    comparison = compare_results(py_result, r_result)
    
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
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("SecActPy core ridge matches SecAct exactly.")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 70)
    
    # Show sample output comparison
    print("\n6. Sample zscore comparison (first 10 rows):")
    print("\nPython:")
    print(py_result['zscore'].head(10))
    print("\nR reference:")
    print(r_result['zscore'].head(10))
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
