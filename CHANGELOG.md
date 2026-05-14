# Changelog

All notable changes to SecActPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-14

### Added
- **`cuda_native` ridge backend**: ctypes wrapper around RidgeCuda's compiled
  CUDA kernel (`libridgecuda_native.so`, vendored at `secactpy/_libs/`). One
  `cudaLaunchKernel` for the full perm sweep — ~14× faster at small `m` than
  the per-iter CuPy path. `resolve_backend('auto')` prefers
  `cuda_native > cupy > numpy` when the library is present.
  - Real sparse path via `cusparseSpMM` (`ridge_cuda_sparse`); no host
    densify, no CuPy fallback. In-kernel column normalization
    (`β = (β_raw − c⊗μ)/σ`) means `col_center=True, col_scale=True` is
    handled in C; older `.so` builds without the sparse symbol fall
    back to CuPy automatically.
  - C-side `build_inv_perm_table_srand()` for `rng_method='srand'`:
    byte-identical to the Python `CStdlibRNG` builder, ~200× faster
    (<50 ms vs ~11 s at `n=8141, n_rand=1000`). End-to-end
    `cuda_native` is 5.6× faster than `cupy` on the GSE100093 fixture
    while staying bit-equivalent on β/SE/z/p.
- **Dash web app** (`secactpy-app` CLI, optional `[app]` extra):
  - **Spatial** tab — Visium / CosMx / Xenium upload, SecAct inference,
    visualization via `secactpy.visualization` and `spatial-gpu` I/O.
  - **Single-cell** tab — pseudo-bulk inference and activity plots.
  - **Bulk** tab — activity change and Kaplan-Meier cohort survival.
  - Cache eviction on each new upload, button disable during inference,
    temp-file cleanup after upload reads.
- **`secactpy.visualization`** (optional `[viz]` extra): nine SecAct-specific
  plotly functions — `activity_distribution`, `celltype_activity_boxplot`,
  `activity_correlation`, `gene_expression_stats`,
  `celltype_expression_boxplot`, `celltype_distribution`, `spatial_density`,
  `activity_change_bar`, `risk_lollipop`. Re-exported from package root.
- **`secactpy.downstream`** (optional `[downstream]` extra): post-inference
  analyses mirroring R `SecAct/R/downstream.R`.
  - `coxph_regression` — delegates to `spatial-gpu`'s
    `secact_coxph_regression` when available; standalone `lifelines`
    fallback otherwise.
  - `signaling_pattern` / `signaling_pattern_gene` — NMF pattern
    discovery; delegates to `spatial-gpu` or falls back to sklearn
    NMF. KDTree replaces `O(n²)` `cdist` in the standalone path.
  - `ccc_scrnaseq` — bulk scRNA-seq cell-cell communication
    (SecActpy-unique, no spatial-gpu equivalent).
  - `ccc_spatial` — thin wrapper around `spatial-gpu`'s
    `secact_spatial_ccc` (spatial-gpu required, no standalone fallback).
- `logistic_regression` and `logit` (from `secactpy.glm`) now exported
  from the package root.
- Bulk vignette replication examples: `bulkChange` and `bulkCohort`.

### Changed
- **Docker / R stack**: Replaced legacy `beibeiru/RidgeR` (now archived) with
  optional accelerators `data2intelligence/RidgeFast` (CPU, cross-platform)
  and `data2intelligence/RidgeCuda` (GPU, Linux+NVIDIA only).
  - `INSTALL_R=true` now installs **SecAct + RidgeFast** by default. The
    `with-r` image gains RidgeFast; the `gpu-with-r` image gains
    RidgeFast + RidgeCuda. The legacy RidgeR is no longer installed in
    any image.
  - New build args `INSTALL_RIDGEFAST` and `INSTALL_RIDGECUDA` (default
    `auto`) let users force-disable the accelerators to test SecAct's
    pure-R fallback.
  - `apptainer/build_sif.sh` verification now checks for `RidgeFast`
    (cpu-r) or `RidgeFast` + `RidgeCuda` (gpu-r) instead of `RidgeR`.
  - Documentation: added native R install matrix for Linux / macOS /
    Windows in `docs/installation.md`
    (`#native-r-install-linux-macos-windows`).
  - Docstrings across `secactpy/` updated from "RidgeR" →
    "R SecAct" / "RidgeFast" where appropriate. Behavior is unchanged.
- New build arg `CUPY_PACKAGE` decouples the CuPy version from the
  hardcoded `cupy-cuda11x`, so a future CUDA base-image bump to 12.x
  only needs `--build-arg CUPY_PACKAGE=cupy-cuda12x`.
- `rng_method` default flipped from `None`
  (`use_gsl_rng=True → CStdlibRNG`) to explicit `'srand'` across all
  ridge entry points (`ridge`, `ridge_with_precomputed_T`, the four
  high-level inference functions, `ridge_batch`, and the three
  streaming entry points). Behaviorally equivalent by default; matches
  RidgeFast / RidgeCuda alignment. `use_gsl_rng` kept as legacy
  fallback when `rng_method=None` is passed explicitly.
- `ridge` NumPy backend now auto-picks Y-row vs T-col permutation by
  `m` vs `p`: `m < p` permutes Y rows
  (`β = T @ Y[fwd_perm[i], :]`), else permutes T columns (existing
  path). 3.3× speedup on the GSE100093 fixture (`m=17, p=1248`).
  Operand-order difference moves cross-impl drift from bit-identical
  to ulp-level (still within 1e-10 tolerance).
- `ridge` CuPy backend: perm-table H2D copy hoisted out of the
  permutation loop; per-batch `mempool.free_all_blocks()` dropped
  (final cleanup only). Output bit-identical.
- Streaming inference: pass 2 (cross-term) and pass 3 (inference) now
  share a single H5AD read (3 reads → 2). `normalize_chunk` and
  `accumulate` vectorized; H5AD string decoding via `np.char.decode`.
  `_free_gpu_memory()` and `_format_ridge_results()` helpers
  consolidate 15 duplicated blocks across `ridge` / `batch` /
  `streaming`.
- `resolve_backend()` extracted in `ridge.py` (dedupes 3 inline
  blocks); `_validate_batch_inputs()` in `batch.py`;
  `_load_sig_matrix()` in `inference.py`; `_get_h5_index()` and
  `_read_h5_sparse_matrix()` in `cli.py`. Type hints modernized
  (`List`/`Tuple` → `list`/`tuple`).
- `rng.py` perm-table cache moved from a hardcoded path to
  XDG-compliant `~/.cache/secactpy/`.
- CI: GitHub Actions bumped to Node 24-compatible majors
  (`actions/checkout v6`, `setup-python v6`, `upload-artifact v7`,
  `download-artifact v8`, `docker/build-push-action v7`,
  `docker/login-action v4`, `docker/setup-buildx-action v4`,
  `peter-evans/dockerhub-description v5`).

### Fixed
- `Dockerfile`: added `libuv1-dev` so the R `fs` package builds from
  source (uncovered by the `cpu-with-r` build). Without it, `fs` and
  six transitive CRAN deps (`networkD3`, `scatterpie`, `shiny`,
  `plotly`, `DT`, `factoextra`) failed to install.
- `H5ADChunkReader`: `read_obs_names()` / `read_var_names()` handle
  H5AD files where the index column name is stored in
  `obs.attrs['_index']` (common in large consortium datasets like the
  Inflammation Atlas). Added `"symbol"` to the gene-column fallback
  list; negative categorical codes handled in `read_var_column()`;
  vectorized categorical reconstruction in `read_obs_column()`.
- Streaming: replaced `Y_chunk.T @ row_means` with
  `row_means @ Y_chunk` to avoid an unnecessary CSC→CSR copy on every
  chunk.
- `secactpy.visualization.activity_correlation`: first-subplot
  annotation now uses `x domain` instead of `x1 domain` for the
  correct xref.
- Dash app: temp files from uploaded payloads are now cleaned up after
  the read; spatial callback no longer imports the unused
  `UI_COLORS`.
- `secactpy.glm`: renamed Fisher information matrix `I`/`I_inv` →
  `info`/`info_inv` to avoid the visually ambiguous `I` (ruff E741);
  documents intent explicitly.

### Migration notes
- Reference H5AD files under `dataset/output/signature/*` were
  generated with legacy RidgeR. They remain numerically valid
  (RidgeFast matches RidgeR to better than `2e-14`), but to fully
  switch the source of truth, re-run
  `sbatch scripts/regenerate_r_reference.sh` against the new image
  once it's published. The script now installs RidgeFast
  automatically.
- The R reference fixture under `tests/` was regenerated on Biowulf
  (R 4.5.2, glibc 2.28) — the previous fixture was generated on a
  different platform with a different `rand()` implementation, which
  caused SE/zscore/pvalue mismatches on test machines. All 37 tests
  now pass with exact numerical agreement (`SE` max diff
  `9.99e-17`).

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
- `from .ridge import ridge_batch` in `inference.py` — `ridge_batch` is
  defined in `batch.py`, not `ridge.py`. This caused `ImportError` when
  calling `secact_activity_inference()` or `secact_activity_inference_st()`
  with `batch_size` set.

## [0.2.1] - 2026-02-08

### Added
- Streaming output (`output_path`, `output_compression`) in all high-level
  inference functions: `secact_activity_inference()`,
  `secact_activity_inference_scrnaseq()`, and `secact_activity_inference_st()`
- `use_gsl_rng` parameter in `ridge_batch()` — enables the ~70x faster NumPy
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

[0.2.5]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.5
[0.2.4]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.4
[0.2.3]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.3
[0.2.2]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.2
[0.2.1]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.1
[0.2.0]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.2.0
[0.1.2]: https://github.com/data2intelligence/SecActPy/releases/tag/v0.1.2
