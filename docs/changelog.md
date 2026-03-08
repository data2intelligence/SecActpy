# Changelog

All notable changes to SecActPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.5] - 2026-02-26

### Added
- **Streaming H5AD processing** for datasets with >5M cells that exceed available
  RAM. Two-pass chunk-reading algorithm via h5py reads CSR rows from H5AD without
  loading the full matrix. Pass 1 accumulates row/column statistics; pass 2
  performs inference in chunks. Peak memory reduced from ~200 GB to ~3 GB for
  5M-cell datasets. Results are numerically identical to the non-streaming path.
  - `streaming=True` and `streaming_chunk_size=50_000` parameters on
    `secact_activity_inference_scrnaseq()` and `secact_activity_inference_st()`
  - New `H5ADChunkReader` class for memory-efficient H5AD chunk reading
  - New `ridge_batch_streaming()` orchestrator for two-pass streaming inference

### Fixed
- `H5ADChunkReader.read_obs_names()` and `read_var_names()` now handle H5AD
  files where the index column name is stored in `obs.attrs['_index']` (e.g.,
  `'cellID'`) rather than as a literal `_index` dataset. This is common in
  large consortium datasets like the Inflammation Atlas.

## [0.2.4] - 2026-02-20

### Added
- `col_center` and `col_scale` parameters for independent control of sparse
  in-flight column normalization in `ridge_batch()`.

## [0.2.3] - 2026-02-15

### Added
- `rng_method` parameter for explicit RNG backend selection (`'srand'`, `'gsl'`,
  `'numpy'`) on all high-level inference functions.
- `is_group_sig=True` as the default (previously `False`).

## [0.2.2] - 2026-02-08

### Added
- `sparse_mode=True` parameter in `ridge()`, `ridge_batch()`, and all
  high-level inference functions for memory-efficient processing of sparse Y
  matrices. Uses `(Y.T @ T.T).T` to compute `T @ Y` without densifying Y,
  with column z-scoring applied as corrections on the small output matrix.
- End-to-end sparse pipeline in `secact_activity_inference_scrnaseq()` and
  `secact_activity_inference_st()`: when `sparse_mode=True`, CPM normalization
  and log2 transform are applied directly on sparse matrices (both are
  zero-preserving), bypassing the dense `secact_activity_inference()` path.
- `row_center=True` parameter in `ridge_batch()` for in-flight row-mean
  centering without densifying Y. Computes row-centered column statistics
  from sparse Y analytically and applies `T @ row_means` correction per
  permutation.

### Fixed
- `from .ridge import ridge_batch` in `inference.py` -- `ridge_batch` is
  defined in `batch.py`, not `ridge.py`. This caused `ImportError` when
  calling `secact_activity_inference()` or `secact_activity_inference_st()`
  with `batch_size` set.

## [0.2.1] - 2026-02-08

### Added
- Streaming output (`output_path`, `output_compression`) in all high-level
  inference functions: `secact_activity_inference()`,
  `secact_activity_inference_scrnaseq()`, and `secact_activity_inference_st()`
- `use_gsl_rng` parameter in `ridge_batch()` -- enables the ~70x faster NumPy
  RNG path for batch processing (previously hardcoded to GSL RNG)

### Fixed
- `use_gsl_rng` was accepted by `secact_activity_inference` but silently
  ignored by `ridge_batch`, which always used the slower GSL RNG. Now
  `ridge_batch` (both dense and sparse paths) respects the flag.

### Changed
- Expanded README batch processing documentation: explains what batch
  processing is, in-memory vs streaming modes, dense vs sparse handling,
  and includes downloadable example data

## [0.2.0] - 2025-01-06

### Changed
- **Official Release**: Migrated to `data2intelligence`
- **PyPI Package**: Now available via `pip install secactpy`
- Updated all documentation and URLs to point to official repository
- Docker images now published to `psychemistz/secactpy`

### Added
- Comprehensive CI/CD pipeline with GitHub Actions
- Automated PyPI publishing on releases
- Automated Docker image builds (CPU, GPU, with-R variants)
- Enhanced test suite covering all major functionality

## [0.1.2] - 2024-12-XX

### Added
- Ridge regression with permutation-based significance testing
- GPU acceleration via CuPy backend (9-34x speedup)
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

[0.2.5]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.5
[0.2.4]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.4
[0.2.3]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.3
[0.2.2]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.2
[0.2.1]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.1
[0.2.0]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.0
[0.1.2]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.1.2
