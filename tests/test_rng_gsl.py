#!/usr/bin/env python3
"""
Step-by-step comparison of Python GSL RNG with R's GSL.

Run this script and compare output with R's GSL output.

Usage:
    python tests/test_gsl_debug.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy.rng import (
    GSLRNG, 
    MT19937Pure,
    get_cached_inverse_perm_table,
)


def test_caching():
    """Test permutation caching functions."""
    print("\n" + "=" * 70)
    print("PERMUTATION CACHING TESTS")
    print("=" * 70)
    
    # Test caching functionality
    print("\n[1] Caching integration:")
    n_genes = 449  # Typical common gene count
    
    import time
    t0 = time.time()
    cached_table = get_cached_inverse_perm_table(
        n=n_genes,
        n_perm=1000,
        seed=0,
        verbose=True
    )
    elapsed = time.time() - t0
    
    print(f"    Cached table shape: {cached_table.shape}")
    print(f"    Time (first call): {elapsed:.3f}s")
    
    # Verify valid permutations
    is_valid = all(sorted(cached_table[i]) == list(range(n_genes)) for i in range(min(10, 1000)))
    print(f"    Valid permutations: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Second call should be faster (cached)
    t0 = time.time()
    cached_table2 = get_cached_inverse_perm_table(n=n_genes, n_perm=1000, seed=0, verbose=False)
    elapsed2 = time.time() - t0
    print(f"    Time (cached): {elapsed2:.3f}s")
    
    # Verify same result
    is_same = np.array_equal(cached_table, cached_table2)
    print(f"    Cached consistency: {'✓ PASS' if is_same else '✗ FAIL'}")
    
    print("\n" + "=" * 70)
    print("CACHING TESTS COMPLETE")
    print("=" * 70)


def main():
    print("=" * 70)
    print("Python GSL MT19937 - Step by Step Debug")
    print("=" * 70)
    
    # ==========================================================================
    # 1. Raw MT19937 output (first 10 values with seed=0)
    # ==========================================================================
    print("\n[1] First 10 raw MT19937 values (seed=0):")
    mt = MT19937Pure(0)
    for i in range(10):
        val = mt.genrand_int32()
        print(f"    {i}: {val}")
    
    # ==========================================================================
    # 2. uniform_int output 
    # ==========================================================================
    print("\n[2] uniform_int(n) for various n (seed=0):")
    rng = GSLRNG(0)
    
    # Test with n=10
    print("    uniform_int(10), first 10 values:")
    vals = [rng.uniform_int(10) for _ in range(10)]
    print(f"    {vals}")
    
    # Reset and test with n=7720 (actual data size)
    rng.reset(0)
    print("\n    uniform_int(7720), first 10 values:")
    vals = [rng.uniform_int(7720) for _ in range(10)]
    print(f"    {vals}")
    
    # ==========================================================================
    # 3. Single shuffle of [0..9]
    # ==========================================================================
    print("\n[3] Fisher-Yates shuffle of [0,1,2,...,9] (seed=0):")
    rng = GSLRNG(0)
    arr = np.arange(10, dtype=np.int32)
    print(f"    Before: {arr.tolist()}")
    rng.shuffle_inplace(arr)
    print(f"    After:  {arr.tolist()}")
    
    # ==========================================================================
    # 4. Cumulative shuffles (as used in permutation table)
    # ==========================================================================
    print("\n[4] Cumulative shuffles of [0..9], 5 permutations (seed=0):")
    rng = GSLRNG(0)
    arr = np.arange(10, dtype=np.int32)
    for i in range(5):
        rng.shuffle_inplace(arr)
        print(f"    Perm {i}: {arr.tolist()}")
    
    # ==========================================================================
    # 5. Permutation table for actual data size
    # ==========================================================================
    print("\n[5] Permutation table (n=7720, n_perm=3, seed=0):")
    rng = GSLRNG(0)
    arr = np.arange(7720, dtype=np.int32)
    
    for perm_idx in range(3):
        rng.shuffle_inplace(arr)
        print(f"    Perm {perm_idx} first 10: {arr[:10].tolist()}")
        print(f"    Perm {perm_idx} last 10:  {arr[-10:].tolist()}")
    
    # ==========================================================================
    # 6. Test caching functions
    # ==========================================================================
    test_caching()
    
    # ==========================================================================
    # Print R code for comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("R CODE TO COMPARE (run in R with RidgeR loaded)")
    print("=" * 70)
    
    r_code = '''
# Run 
library(RidgeR)
.Call("debug_gsl_rng", PACKAGE = "RidgeR")
'''
    print(r_code)
    print("=" * 70)


if __name__ == "__main__":
    main()
