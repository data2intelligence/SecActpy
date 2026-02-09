# GPU Acceleration

SecActPy supports GPU acceleration via CuPy for large-scale analysis.

## Setup

```python
from secactpy import secact_activity_inference, CUPY_AVAILABLE

print(f"GPU available: {CUPY_AVAILABLE}")

# Auto-detect GPU
result = secact_activity_inference(expression, backend='auto')

# Force GPU
result = secact_activity_inference(expression, backend='cupy')
```

## Performance Benchmarks

| Dataset | R (Mac M1) | R (Linux) | Py (CPU) | Py (GPU) | Speedup |
|---------|------------|-----------|----------|----------|---------|
| Bulk (1,170 sp × 1,000 samples) | 74.4s | 141.6s | 128.8s | 6.7s | 11–19x |
| scRNA-seq (1,170 sp × 788 cells) | 54.9s | 117.4s | 104.8s | 6.8s | 8–15x |
| Visium (1,170 sp × 3,404 spots) | 141.7s | 379.8s | 381.4s | 11.2s | 13–34x |
| CosMx (151 sp × 443,515 cells) | 936.9s | 976.1s | 1226.7s | 99.9s | 9–12x |

<details>
<summary>Benchmark Environment</summary>

- **Mac CPU**: M1 Pro with VECLIB (8 cores)
- **Linux CPU**: AMD EPYC 7543P (4 cores)
- **Linux GPU**: NVIDIA A100-SXM4-80GB

</details>

## CUDA Version Notes

```bash
# CUDA 11.x
pip install "secactpy[gpu]"

# CUDA 12.x (do NOT use [gpu] extra)
pip install secactpy
pip install cupy-cuda12x
```

> **Important (CUDA 12.x users)**: Do **not** use the `[gpu]` extra on CUDA 12.x systems — it installs `cupy-cuda11x`, which conflicts with `cupy-cuda12x`. If you already installed with `[gpu]`, remove the conflicting package first:
> ```bash
> pip uninstall cupy-cuda11x
> pip install cupy-cuda12x
> ```
