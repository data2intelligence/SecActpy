# SecActPy

**Secreted Protein Activity Inference using Ridge Regression**

[![PyPI version](https://badge.fury.io/py/secactpy.svg)](https://pypi.org/project/secactpy/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/data2intelligence/SecActPy/actions/workflows/tests.yml/badge.svg)](https://github.com/data2intelligence/SecActPy/actions/workflows/tests.yml)
[![Docker](https://img.shields.io/docker/pulls/psychemistz/secactpy)](https://hub.docker.com/r/psychemistz/secactpy)

Python implementation of [SecAct](https://github.com/data2intelligence/SecAct) for inferring secreted protein activities from gene expression data.

## Key Features

- **SecAct Compatible**: Matches R SecAct/RidgeR results on the same platform (`rng_method='srand'`)
- **GPU Acceleration**: Optional CuPy backend for large-scale analysis
- **Million-Sample Scale**: Batch processing with streaming output for massive datasets
- **Built-in Signatures**: Includes SecAct and CytoSig signature matrices
- **Multi-Platform Support**: Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics (Visium, CosMx)
- **Smart Caching**: Optional permutation table caching for faster repeated analyses
- **Sparse-Aware**: Automatic memory-efficient processing for sparse single-cell data

## Getting Started

- [Installation](installation.md) — install SecActPy from PyPI or GitHub
- [Quick Start](quickstart.md) — run your first analysis in minutes

## User Guide

- [Batch Processing](batch_processing.md) — handle large datasets with memory-efficient batching
- [GPU Acceleration](gpu_acceleration.md) — speed up computation with CuPy
- [Reproducibility](reproducibility.md) — RNG backends for cross-platform reproducibility
- [Docker](docker.md) — run SecActPy in Docker containers
- [CLI Reference](cli.md) — command-line interface documentation

## API Reference

- [API Reference](api_reference.md) — full function signatures and parameters
- [Advanced API](advanced_api.md) — low-level `ridge()` / `ridge_batch()` usage

## Citation

If you use SecActPy in your research, please cite:

> Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang.
> **Inference of secreted protein activities in intercellular communication.**
> [GitHub: data2intelligence/SecAct](https://github.com/data2intelligence/SecAct)

## Related Projects

- [SecAct](https://github.com/data2intelligence/SecAct) — Original R implementation
- [RidgeR](https://github.com/beibeiru/RidgeR) — R ridge regression package
- [SpaCET](https://github.com/data2intelligence/SpaCET) — Spatial transcriptomics cell type analysis
- [CytoSig](https://github.com/data2intelligence/CytoSig) — Cytokine signaling inference

## License

MIT License — see [LICENSE](https://github.com/data2intelligence/SecActpy/blob/main/LICENSE) for details.
