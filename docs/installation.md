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
