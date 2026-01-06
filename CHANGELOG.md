# Changelog

All notable changes to SecActPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-06

### Changed
- **Official Release**: Migrated to `data2intelligence` organization
- **PyPI Package**: Now available via `pip install secactpy`
- Updated all documentation and URLs to point to official repository
- Docker images now published to `data2intelligence/secactpy`

### Added
- Comprehensive CI/CD pipeline with GitHub Actions
- Automated PyPI publishing on releases
- Automated Docker image builds (CPU, GPU, with-R variants)
- Enhanced test suite covering all major functionality

## [0.1.2] - 2024-12-XX

### Added
- Ridge regression with permutation-based significance testing
- GPU acceleration via CuPy backend (9–34x speedup)
- Batch processing with streaming H5AD output for million-sample datasets
- Automatic sparse matrix handling in `ridge_batch()`
- Built-in SecAct and CytoSig signature matrices
- GSL-compatible RNG for R/RidgeR reproducibility
- Support for Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics
- Cell type resolution for ST data (`cell_type_col`, `is_spot_level`)
- Optional permutation table caching (`use_cache`)
- Command-line interface for common workflows
- Docker support with CPU, GPU, and R variants

### Features
- **High-Level API**:
  - `secact_activity_inference()` - Bulk RNA-seq inference
  - `secact_activity_inference_st()` - Spatial transcriptomics inference
  - `secact_activity_inference_scrnaseq()` - scRNA-seq inference

- **Core API**:
  - `ridge()` - Single-call ridge regression
  - `ridge_batch()` - Batch processing for large datasets
  - `load_signature()` - Load built-in signature matrices

- **Performance**:
  - GPU acceleration achieving 9-34x speedup
  - Memory-efficient sparse matrix processing
  - Streaming output for very large datasets

- **Compatibility**:
  - Produces identical results to R SecAct/RidgeR
  - GSL-compatible random number generator
  - Cross-platform support (Linux, macOS, Windows)

[0.2.0]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.0
[0.1.2]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.1.2
