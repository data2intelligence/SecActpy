# Docker Usage Guide

This guide explains how to use SecActPy with Docker, including CPU, GPU, and R-enabled versions.

## Quick Start

### Pull Pre-built Images

```bash
# CPU version (Python only) - recommended for most users
docker pull psychemistz/secactpy:latest

# GPU version (Python + CuPy)
docker pull psychemistz/secactpy:gpu

# CPU + R version (Python + SecAct/RidgeR/SpaCET)
docker pull psychemistz/secactpy:with-r

# GPU + R version (full stack)
docker pull psychemistz/secactpy:gpu-with-r
```

### Build Images Locally

```bash
# CPU version (default, Python only ~5 min)
docker build -t secactpy:latest .

# GPU version (Python + CuPy ~10 min)
docker build -t secactpy:gpu --build-arg USE_GPU=true .

# CPU + R version (Python + R packages ~30-60 min)
docker build -t secactpy:with-r --build-arg INSTALL_R=true .

# GPU + R version (full stack ~40-90 min)
docker build -t secactpy:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Set to `true` for GPU/CUDA support |
| `INSTALL_R` | `false` | Set to `true` to include R and packages |

## Available Docker Images

| Tag | Python | GPU | R | Size | Use Case |
|-----|--------|-----|---|------|----------|
| `latest` / `cpu` | ✓ | ✗ | ✗ | ~2 GB | General use |
| `gpu` | ✓ | ✓ | ✗ | ~6 GB | GPU acceleration |
| `with-r` / `cpu-with-r` | ✓ | ✗ | ✓ | ~3.5 GB | R cross-validation |
| `gpu-with-r` | ✓ | ✓ | ✓ | ~8 GB | Full stack |

## R Packages Included

When building with `INSTALL_R=true`, the following packages are installed. CRAN packages are fetched as pre-compiled binaries from the [Posit Public Package Manager](https://packagemanager.posit.co/) for faster builds.

### From CRAN (pre-compiled binaries)

**Core:**
- remotes, BiocManager, devtools
- Matrix, Rcpp, RcppArmadillo, RcppEigen
- ggplot2, dplyr, tidyr, data.table, httr, jsonlite, R6, crayon

**SecAct dependencies:**
- reshape2, patchwork, NMF, akima, gganimate, metap
- circlize, ggalluvial, networkD3, survival, survminer
- ggpubr, car, lme4, sp

**SpaCET dependencies:**
- scatterpie, png, shiny, plotly, DT
- factoextra, NbClust, cluster, pbmcapply, psych
- arrow, RANN, sctransform

### From Bioconductor (`BiocManager::install()`)
- Biobase, S4Vectors, IRanges
- SummarizedExperiment, SingleCellExperiment, rhdf5
- ComplexHeatmap, limma, UCell, BiRewire

### From GitHub (`remotes::install_github()`)
- **RidgeR**: `beibeiru/RidgeR` — Ridge regression with GSL RNG
- **SecAct**: `data2intelligence/SecAct` — Secreted protein activity inference
- **SpaCET**: `data2intelligence/SpaCET` — Spatial transcriptomics cell type analysis
- **MUDAN**: `JEFworks/MUDAN` — Multi-scale Diffusion for Uniform Manifold Approximation (SpaCET dependency)

## Running Containers

### Interactive Mode

```bash
# CPU
docker run -it --rm -v $(pwd):/workspace psychemistz/secactpy:latest

# GPU
docker run -it --rm --gpus all -v $(pwd):/workspace psychemistz/secactpy:gpu

# CPU + R (for cross-validation)
docker run -it --rm -v $(pwd):/workspace psychemistz/secactpy:with-r
```

### Run Scripts

```bash
# Python script
docker run --rm -v $(pwd):/workspace psychemistz/secactpy:latest python your_script.py

# R script
docker run --rm -v $(pwd):/workspace psychemistz/secactpy:with-r Rscript your_script.R
```

## Using Docker Compose

Docker Compose simplifies container management:

```bash
# Start CPU container
docker-compose up -d secactpy

# Start GPU container
docker-compose up -d secactpy-gpu

# Start CPU + R container
docker-compose up -d secactpy-r

# Enter a running container
docker-compose exec secactpy bash

# Stop all containers
docker-compose down
```

### Available Services

| Service | GPU | R | Port | Description |
|---------|-----|---|------|-------------|
| `secactpy` | ✗ | ✗ | - | CPU interactive |
| `secactpy-gpu` | ✓ | ✗ | - | GPU interactive |
| `secactpy-r` | ✗ | ✓ | - | CPU + R interactive |
| `secactpy-gpu-r` | ✓ | ✓ | - | GPU + R interactive |
| `secactpy-jupyter` | ✗ | ✗ | 8888 | Jupyter Lab (CPU) |
| `secactpy-jupyter-gpu` | ✓ | ✗ | 8889 | Jupyter Lab (GPU) |
| `secactpy-jupyter-r` | ✗ | ✓ | 8890 | Jupyter Lab (CPU + R) |

### Jupyter Lab

```bash
# Start Jupyter (CPU)
docker-compose up secactpy-jupyter
# Open http://localhost:8888

# Start Jupyter (GPU)
docker-compose up secactpy-jupyter-gpu
# Open http://localhost:8889

# Start Jupyter (CPU + R)
docker-compose up secactpy-jupyter-r
# Open http://localhost:8890
```

## Validation Examples

### Verify Python Installation

```bash
docker run --rm psychemistz/secactpy:latest python -c "
import secactpy
print(f'SecActPy {secactpy.__version__}')
print(f'GPU available: {secactpy.CUPY_AVAILABLE}')

# Load signature
sig = secactpy.load_signature('secact')
print(f'Signature: {sig.shape}')
"
```

### Verify R Installation

```bash
docker run --rm psychemistz/secactpy:with-r Rscript -e "
cat('R version:', R.version.string, '\n')

# Check packages
for (pkg in c('SecAct', 'RidgeR', 'SpaCET', 'Biobase')) {
    if (require(pkg, quietly = TRUE, character.only = TRUE)) {
        cat(pkg, 'OK -', as.character(packageVersion(pkg)), '\n')
    } else {
        cat(pkg, 'NOT FOUND\n')
    }
}
"
```

### Cross-Validation (R vs Python)

```bash
# Enter container with R
docker run -it --rm -v $(pwd):/workspace psychemistz/secactpy:with-r

# Inside container - Run R inference
Rscript -e "
library(SecAct)
data <- read.table('your_data.txt', row.names=1, header=TRUE)
result <- SecAct.inference.gsl(data)
write.table(result\$zscore, 'r_zscore.txt', quote=FALSE)
"

# Run Python inference
python -c "
import pandas as pd
from secactpy import secact_activity_inference

data = pd.read_csv('your_data.txt', sep='\t', index_col=0)
result = secact_activity_inference(data, is_differential=True)
result['zscore'].to_csv('py_zscore.txt', sep='\t')
"

# Compare results
python -c "
import pandas as pd
import numpy as np

r_result = pd.read_csv('r_zscore.txt', sep='\t', index_col=0)
py_result = pd.read_csv('py_zscore.txt', sep='\t', index_col=0)

diff = np.abs(r_result.values - py_result.values)
print(f'Max difference: {diff.max():.2e}')
print(f'Mean difference: {diff.mean():.2e}')
"
```

## GPU Support

### Prerequisites

1. NVIDIA GPU with CUDA support
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Installation (Ubuntu)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Check GPU is accessible
docker run --rm --gpus all psychemistz/secactpy:gpu nvidia-smi

# Check CuPy
docker run --rm --gpus all psychemistz/secactpy:gpu python -c "
import cupy as cp
print(f'CuPy version: {cp.__version__}')
x = cp.arange(10)
print(f'GPU array: {x}')
"
```

## Performance Benchmarks

Performance comparison using Ly86 dataset (16,325 genes × 100 samples, 1,000 permutations).

### R Backends (RidgeR)

| Backend | macOS Native | Docker (BLAS=1) | Docker (BLAS=8) |
|---------|-------------|-----------------|-----------------|
| gsl.old | 31.2s | 592.5s | 417.8s |
| Yrow.st | 26.3s | 592.2s | 428.5s |
| Tcol.st | 81.2s | 636.3s | 503.1s |
| Tcol.mt(8) | 37.6s | **384.3s** | 466.3s |
| Yrow.mt(8) | **25.6s** | 501.1s | **386.8s** |

### Python (SecActPy)

| Environment | srand | gsl |
|-------------|-------|-----|
| macOS Native | 52.6s | 56.8s |
| Docker (BLAS=1) | 386.1s | 387.0s |

### Best Elapsed Time per Environment

| Environment | Best R | Best Python |
|-------------|--------|-------------|
| macOS native | **25.6s** (Yrow.mt) | **52.6s** (srand) |
| Docker | **384.3s** (Tcol.mt) | **386.1s** (srand) |

### Notes

- **macOS native** uses Apple vecLib/Accelerate with AMX coprocessor (dedicated matrix multiplication hardware), explaining the ~15x speed advantage over Docker
- **Docker on macOS** uses generic OpenBLAS with basic ARM NEON SIMD (no Apple Silicon optimization)
- **Docker on native Linux servers** (x86_64) has <1% overhead vs native — both use the same OpenBLAS/MKL, so the macOS-specific slowdown does not apply
- All backends produce **numerically identical** z-scores (max|diff| < 2e-14)

### Optimal Docker Configuration

For best Docker performance, set `OPENBLAS_NUM_THREADS=1` and use multi-threaded RidgeR backends:

```bash
docker run --rm -e OPENBLAS_NUM_THREADS=1 \
  -v $(pwd):/workspace psychemistz/secactpy:with-r \
  Rscript -e "library(RidgeR); SecAct.inference.Tcol.mt(expr, ncores=8)"
```

This avoids thread oversubscription (OpenBLAS threads × OpenMP threads > CPU cores) which can cause severe performance degradation.

## Troubleshooting

### R Package Installation Failures

If R packages fail to install, try building with verbose output:

```bash
docker build -t secactpy:with-r --build-arg INSTALL_R=true --progress=plain .
```

Common issues:
- **Network timeout**: Increase timeout in Dockerfile or retry
- **Missing system library**: Add to apt-get install list
- **GitHub rate limit**: Wait and retry, or use a GitHub token

### Permission Issues

```bash
# Run as current user
docker run -it --rm -u $(id -u):$(id -g) -v $(pwd):/workspace psychemistz/secactpy:latest
```

### Memory Issues

```bash
# Increase memory limit
docker run -it --rm -m 16g -v $(pwd):/workspace psychemistz/secactpy:latest
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker info | grep -i gpu

# Verify nvidia-container-toolkit
which nvidia-container-toolkit
```

## Customization

### Add Your Own Packages

Create a custom Dockerfile:

```dockerfile
FROM psychemistz/secactpy:with-r

# Add R packages
RUN R -e "install.packages('your_r_package', repos='https://cloud.r-project.org/')"
RUN R -e "BiocManager::install('bioc_package')"
RUN R -e "remotes::install_github('user/repo')"

# Add Python packages
RUN pip3 install your_python_package
```

### Mount Additional Data

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/data:/data:ro \
  -v /path/to/results:/results \
  psychemistz/secactpy:latest
```

## Singularity / HPC

On HPC clusters where Docker is not available, use Singularity or Apptainer to pull the Docker images:

```bash
# Load Singularity or Apptainer (cluster-specific)
module load singularity   # or: module load apptainer

# Pull CPU image
singularity pull docker://psychemistz/secactpy:latest

# Pull R-enabled image
singularity pull docker://psychemistz/secactpy:with-r

# Run interactively
singularity exec secactpy_latest.sif python3 -c "import secactpy; print(secactpy.__version__)"

# Run with data binding
singularity exec --bind /path/to/data:/data secactpy_latest.sif python3 your_script.py

# Run R inside the container
singularity exec secactpy_with-r.sif Rscript -e "library(SecAct); library(RidgeR); cat('OK\n')"
```

> **Note:** Building images from the Dockerfile on HPC requires `fakeroot` support. If unavailable, pull the pre-built images from Docker Hub instead.

## CI/CD Notes

- Default CI builds use `INSTALL_R=false` for speed
- R-enabled builds are triggered on releases or manual dispatch
- Use `workflow_dispatch` with `build_r=true` to build R images manually

```yaml
# Trigger R build manually in GitHub Actions
workflow_dispatch:
  inputs:
    build_r:
      description: 'Build images with R packages'
      default: 'true'
```
