# Installation

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- Pandas >= 1.3
- SciPy >= 1.7
- h5py >= 3.0
- anndata >= 0.8
- scanpy >= 1.9

**Optional:** CuPy >= 10.0 (GPU acceleration)

!!! tip "Virtual Environment"
    Create a virtual environment before installing to avoid dependency conflicts:

    ```bash
    python -m venv secactpy-env
    source secactpy-env/bin/activate   # Linux/macOS
    # secactpy-env\Scripts\activate    # Windows
    ```

## From PyPI (Recommended)

```bash
# CPU Only
pip install secactpy

# With GPU Support (CUDA 11.x)
pip install "secactpy[gpu]"

# With GPU Support (CUDA 12.x)
pip install secactpy
pip install cupy-cuda12x
```

## From GitHub

```bash
# CPU Only
pip install git+https://github.com/data2intelligence/SecActpy.git

# With GPU Support (CUDA 11.x)
pip install "secactpy[gpu] @ git+https://github.com/data2intelligence/SecActpy.git"

# With GPU Support (CUDA 12.x)
pip install git+https://github.com/data2intelligence/SecActpy.git
pip install cupy-cuda12x
```

## Development Installation

```bash
git clone https://github.com/data2intelligence/SecActpy.git
cd SecActpy
pip install -e ".[dev]"
```

## Documentation Tools

To build the documentation site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Native R install (Linux / macOS / Windows)

SecActPy is a pure-Python package and runs on its own. The optional R-side
stack (SecAct + accelerators) is needed only when you want to run the original
R implementation alongside SecActPy for cross-validation or benchmarking.

The R packages are:

| Package | Role | Required | Platforms |
|---------|------|----------|-----------|
| **SecAct** (`data2intelligence/SecAct`) | R-native inference (no compiled deps) | Yes | Linux / macOS / Windows |
| **RidgeFast** (`data2intelligence/RidgeFast`) | Optional CPU accelerator (R + C, GSL) | No | Linux / macOS / Windows |
| **RidgeCuda** (`data2intelligence/RidgeCuda`) | Optional GPU accelerator (R + CUDA) | No | Linux + NVIDIA only |

The legacy `beibeiru/RidgeR` package is archived and is not used by SecActPy
anymore.

### Linux (Debian / Ubuntu)

```bash
# 1. R + system libraries (GSL for RidgeFast, OpenBLAS for BLAS, TBB for OpenMP)
sudo apt update
sudo apt install -y r-base r-base-dev libgsl-dev libopenblas-dev libtbb-dev

# 2. R packages
Rscript -e 'install.packages(c("remotes", "BiocManager"))'
Rscript -e 'remotes::install_github(c(
  "data2intelligence/SecAct",
  "data2intelligence/RidgeFast"   # optional CPU accelerator
))'

# 3. (Optional, NVIDIA GPU only) RidgeCuda
# Requires CUDA Toolkit >= 11.0 with cuBLAS / cuSOLVER / cuSPARSE / cuRAND.
Rscript -e 'remotes::install_github("data2intelligence/RidgeCuda")'
```

### macOS

```bash
# 1. Toolchain via Homebrew
brew install r gsl libomp openblas

# 2. R packages
Rscript -e 'install.packages(c("remotes", "BiocManager"))'
Rscript -e 'remotes::install_github(c(
  "data2intelligence/SecAct",
  "data2intelligence/RidgeFast"
))'
```

!!! note
    RidgeCuda is **not supported on macOS** — Apple does not ship NVIDIA
    drivers. Use SecAct (R-native) or SecActPy with `backend="cupy"` from a
    Linux machine for GPU inference.

### Windows

1. Install **R** from <https://cloud.r-project.org> and **Rtools43+** from
   <https://cran.r-project.org/bin/windows/Rtools/> (Rtools provides the
   compiler toolchain plus a bundled `pacman` for system libraries).
2. From the Rtools mingw64 / ucrt64 shell, install GSL:

   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gsl
   ```

3. From a regular R session:

   ```r
   install.packages(c("remotes", "BiocManager"))
   remotes::install_github(c(
     "data2intelligence/SecAct",
     "data2intelligence/RidgeFast"
   ))
   ```

!!! note
    RidgeCuda is **not supported on Windows**. The Linux+NVIDIA combination is
    the only supported configuration for GPU R-side inference.

### Verify the install

```r
library(SecAct)
library(RidgeFast)        # if installed
# library(RidgeCuda)      # GPU only

cat("R version:", R.version.string, "\n")
cat("SecAct:",    as.character(packageVersion("SecAct")),    "\n")
cat("RidgeFast:", as.character(packageVersion("RidgeFast")), "\n")
```

If you skip RidgeFast, SecAct falls back to its pure-R ridge path. That path
is correct but slower; the CPU accelerator is recommended for any non-trivial
workload.
