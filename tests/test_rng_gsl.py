#!/usr/bin/env python3
"""
Step-by-step comparison of Python RNG backends.

Run this script and compare output with R's SecAct output.

Usage:
    python tests/test_rng_gsl.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy.rng import (
    CStdlibRNG,
    GSLRNG,
    NumpyRNG,
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
    print(f"    Valid permutations: {'PASS' if is_valid else 'FAIL'}")

    # Second call should be faster (cached)
    t0 = time.time()
    cached_table2 = get_cached_inverse_perm_table(n=n_genes, n_perm=1000, seed=0, verbose=False)
    elapsed2 = time.time() - t0
    print(f"    Time (cached): {elapsed2:.3f}s")

    # Verify same result
    is_same = np.array_equal(cached_table, cached_table2)
    print(f"    Cached consistency: {'PASS' if is_same else 'FAIL'}")

    print("\n" + "=" * 70)
    print("CACHING TESTS COMPLETE")
    print("=" * 70)


def main():
    print("=" * 70)
    print("SecActPy RNG Backends - Step by Step Debug")
    print("=" * 70)

    # ==========================================================================
    # 1. Raw MT19937 output (first 10 values with seed=0)
    # ==========================================================================
    print("\n[1] First 10 raw MT19937 values (seed=0 -> actual 4357 per GSL):")
    mt = MT19937Pure(4357)
    for i in range(10):
        val = mt.genrand_int32()
        print(f"    {i}: {val}")

    # ==========================================================================
    # 2. CStdlibRNG - matches R SecAct
    # ==========================================================================
    print("\n[2] CStdlibRNG (C stdlib rand) shuffle of [0..9] (seed=0):")
    rng = CStdlibRNG(0)
    arr = np.arange(10, dtype=np.int32)
    print(f"    Before: {arr.tolist()}")
    rng.shuffle_inplace(arr)
    print(f"    After:  {arr.tolist()}")

    # ==========================================================================
    # 3. GSLRNG shuffle
    # ==========================================================================
    print("\n[3] GSLRNG (GSL MT19937) shuffle of [0..9] (seed=0):")
    rng = GSLRNG(0)
    arr = np.arange(10, dtype=np.int32)
    print(f"    Before: {arr.tolist()}")
    rng.shuffle_inplace(arr)
    print(f"    After:  {arr.tolist()}")

    # ==========================================================================
    # 4. NumpyRNG shuffle
    # ==========================================================================
    print("\n[4] NumpyRNG shuffle of [0..9] (seed=0):")
    rng = NumpyRNG(0)
    arr = np.arange(10, dtype=np.int32)
    print(f"    Before: {arr.tolist()}")
    rng.shuffle_inplace(arr)
    print(f"    After:  {arr.tolist()}")

    # ==========================================================================
    # 5. Cumulative shuffles with CStdlibRNG (as used in permutation table)
    # ==========================================================================
    print("\n[5] CStdlibRNG cumulative shuffles of [0..9], 5 permutations (seed=0):")
    rng = CStdlibRNG(0)
    arr = np.arange(10, dtype=np.int32)
    for i in range(5):
        rng.shuffle_inplace(arr)
        print(f"    Perm {i}: {arr.tolist()}")

    # ==========================================================================
    # 6. Permutation table for actual data size
    # ==========================================================================
    print("\n[6] CStdlibRNG permutation table (n=7720, n_perm=3, seed=0):")
    rng = CStdlibRNG(0)
    arr = np.arange(7720, dtype=np.int32)

    for perm_idx in range(3):
        rng.shuffle_inplace(arr)
        print(f"    Perm {perm_idx} first 10: {arr[:10].tolist()}")
        print(f"    Perm {perm_idx} last 10:  {arr[-10:].tolist()}")

    # ==========================================================================
    # 7. Test caching functions
    # ==========================================================================
    test_caching()

    # ==========================================================================
    # Print C code for comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("C CODE TO COMPARE (compile with: gcc -o gen_perm gen_perm.c)")
    print("=" * 70)

    c_code = '''
#include <stdio.h>
#include <stdlib.h>

void shuffle(int array[], const int n) {
    int i, j, tmp;
    for (i = 0; i < n - 1; i++) {
        j = i + rand() / (RAND_MAX / (n - i) + 1);
        tmp = array[j]; array[j] = array[i]; array[i] = tmp;
    }
}

int main() {
    srand(0);
    int array[] = {0,1,2,3,4,5,6,7,8,9};
    int i, p;
    for (p = 0; p < 5; p++) {
        shuffle(array, 10);
        printf("Perm %d: ", p);
        for (i = 0; i < 10; i++) printf("%d ", array[i]);
        printf("\\n");
    }
    return 0;
}
'''
    print(c_code)
    print("=" * 70)


if __name__ == "__main__":
    main()
