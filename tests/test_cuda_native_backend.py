"""SecActpy backend='cuda_native' parity gate.

Asserts equivalence vs backend='cupy' on β / SE / z / p with the same
input + RNG settings. Run on a GPU node before any SecActpy release.

Tolerances:
  β, SE, z   max|Δ| ≤ 1e-10
  p-value    max|Δ| = 0  (integer counts; same perm table → same counts)
"""
import sys
import numpy as np

sys.path.insert(0, "/vf/users/parks34/projects/1ridgesig/SecActpy")
from secactpy.ridge import ridge, CUDA_NATIVE_AVAILABLE, CUPY_AVAILABLE


def main():
    if not (CUDA_NATIVE_AVAILABLE and CUPY_AVAILABLE):
        print(f"SKIP — CUDA_NATIVE_AVAILABLE={CUDA_NATIVE_AVAILABLE} "
              f"CUPY_AVAILABLE={CUPY_AVAILABLE}")
        return

    rng = np.random.default_rng(1)
    n, p, m = 8141, 1248, 17
    X = rng.standard_normal((n, p)); X = (X - X.mean(0)) / X.std(0, ddof=0)
    Y = rng.standard_normal((n, m)); Y = (Y - Y.mean(0)) / Y.std(0, ddof=0)
    lam, nrand = 100.0, 1000

    print(f"Fixture: n={n}, p={p}, m={m}, nrand={nrand}\n")

    print("--- backend='cuda_native' ---")
    r_native = ridge(X, Y, lambda_=lam, n_rand=nrand,
                     seed=0, rng_method="srand", backend="cuda_native",
                     verbose=True)
    print(f"  method={r_native['method']}  time={r_native['time']:.3f}s")

    print("\n--- backend='cupy' ---")
    r_cupy = ridge(X, Y, lambda_=lam, n_rand=nrand,
                   seed=0, rng_method="srand", backend="cupy",
                   verbose=True)
    print(f"  method={r_cupy['method']}  time={r_cupy['time']:.3f}s")

    print(f"\nspeedup (cupy / cuda_native): "
          f"{r_cupy['time'] / r_native['time']:.2f}x\n")

    keys = ["beta", "se", "zscore", "pvalue"]
    tols = {"beta": 1e-10, "se": 1e-10, "zscore": 1e-10, "pvalue": 0.0}
    fail = False
    for k in keys:
        a, b = np.asarray(r_native[k]), np.asarray(r_cupy[k])
        d = float(np.max(np.abs(a - b)))
        ok = d <= tols[k]
        print(f"  {k:8s} max|Δ| = {d:.2e}   "
              f"(require ≤ {tols[k]:.0e})   {'OK' if ok else 'FAIL'}")
        fail |= not ok
    if fail:
        print("\nFAIL — cuda_native diverges from cupy.")
        sys.exit(1)
    print("\nPASS — backend='cuda_native' bit-equivalent to backend='cupy'.")


if __name__ == "__main__":
    main()
