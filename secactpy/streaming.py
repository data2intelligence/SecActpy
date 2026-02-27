"""
Streaming H5AD processing for memory-efficient single-cell inference.

Enables processing of datasets with >5M cells on memory-constrained nodes
by reading H5AD files in chunks via h5py, avoiding full matrix loading.

Two-pass algorithm:
  Pass 1: Accumulate row_sums, col_sums, col_sum_sq from normalized chunks
          → compute row_means, column stats for row-centered data
  Pass 2: Re-read chunks, compute per-chunk cross terms, feed sub-batches
          to existing _process_sparse_batch_{numpy,cupy}()

Results are numerically identical to the non-streaming path.

Usage:
------
    >>> from secactpy import secact_activity_inference_scrnaseq
    >>> result = secact_activity_inference_scrnaseq(
    ...     "large_dataset.h5ad",
    ...     cell_type_col="cell_type",
    ...     is_single_cell_level=True,
    ...     streaming=True,
    ...     streaming_chunk_size=50_000,
    ...     output_path="results.h5ad",
    ...     verbose=True,
    ... )
"""

import gc
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from scipy import sparse as sps

from .batch import (
    StreamingResultWriter,
    _compute_population_stats,
    _compute_projection_components,
    _PopulationStats,
    _process_sparse_batch_cupy,
    _process_sparse_batch_numpy,
)
from .ridge import CUPY_AVAILABLE, DEFAULT_LAMBDA, DEFAULT_NRAND, DEFAULT_SEED, EPS, _get_rng
from .rng import generate_inverse_permutation_table_fast, get_cached_inverse_perm_table

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import cupy as cp
except ImportError:
    cp = None


# =============================================================================
# H5AD Chunk Reader
# =============================================================================


class H5ADChunkReader:
    """Memory-efficient chunk reader for H5AD files via h5py.

    Reads CSR rows from H5AD's sparse X storage without loading the full
    matrix. Supports context manager protocol.

    Parameters
    ----------
    h5ad_path : str or Path
        Path to .h5ad file.
    chunk_size : int
        Number of cells (rows in AnnData, i.e. obs) per chunk.
    """

    def __init__(self, h5ad_path: Union[str, Path], chunk_size: int = 50_000):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for streaming H5AD processing")

        self.path = str(h5ad_path)
        self.chunk_size = chunk_size
        self._file = None
        self._open()

    # -- file lifecycle -------------------------------------------------------

    def _open(self):
        self._file = h5py.File(self.path, "r")
        self._detect_x_location()
        self._read_shape()

    def _detect_x_location(self):
        """Find the sparse X matrix in the H5AD hierarchy."""
        # Prefer raw/X (raw counts) over X (which may be normalized)
        for prefix in ("raw/X", "raw/var", "X"):
            if prefix in self._file:
                break

        # Check for raw counts first
        if "raw" in self._file and "X" in self._file["raw"]:
            self._x_group = self._file["raw/X"]
            self._use_raw = True
        elif "X" in self._file:
            x_node = self._file["X"]
            if isinstance(x_node, h5py.Group):
                self._x_group = x_node
            else:
                # Dense X stored as a dataset — not sparse
                raise ValueError(
                    "H5AD X is stored as dense matrix. "
                    "Streaming mode requires sparse (CSR/CSC) storage."
                )
            self._use_raw = False
        else:
            raise ValueError("Cannot find X matrix in H5AD file")

        # Verify sparse encoding
        def _get_str_attr(group, key):
            if key not in group.attrs:
                return ""
            val = group.attrs[key]
            if isinstance(val, bytes):
                return val.decode()
            return str(val)

        encoding = _get_str_attr(self._x_group, "encoding-type")
        if not encoding:
            encoding = _get_str_attr(self._x_group, "h5sparse_format")

        # Accept csr_matrix or csc_matrix
        if "csr" in encoding:
            self._encoding = "csr"
        elif "csc" in encoding:
            self._encoding = "csc"
        else:
            # Try to detect from dataset names
            if "indptr" in self._x_group and "indices" in self._x_group and "data" in self._x_group:
                shape = self._x_group.attrs.get("shape", None)
                if shape is not None:
                    indptr_len = self._x_group["indptr"].shape[0]
                    if indptr_len == shape[0] + 1:
                        self._encoding = "csr"
                    elif indptr_len == shape[1] + 1:
                        self._encoding = "csc"
                    else:
                        self._encoding = "csr"  # default guess
                else:
                    self._encoding = "csr"
            else:
                raise ValueError(
                    f"Unsupported sparse encoding '{encoding}'. "
                    "Streaming requires CSR or CSC H5AD storage."
                )

    def _read_shape(self):
        """Read matrix shape from H5AD attributes."""
        shape = self._x_group.attrs.get("shape", None)
        if shape is not None:
            self.n_obs, self.n_vars = int(shape[0]), int(shape[1])
        else:
            # Infer from indptr
            indptr_len = self._x_group["indptr"].shape[0]
            nnz = self._x_group["data"].shape[0]
            if self._encoding == "csr":
                self.n_obs = indptr_len - 1
                # n_vars from indices max or var metadata
                self.n_vars = int(self._x_group["indices"][:].max()) + 1 if nnz > 0 else 0
            else:
                self.n_vars = indptr_len - 1
                self.n_obs = int(self._x_group["indices"][:].max()) + 1 if nnz > 0 else 0

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- metadata access ------------------------------------------------------

    def read_var_names(self) -> list:
        """Read gene/variable names from H5AD."""
        if self._use_raw and "raw" in self._file:
            var_group = self._file["raw/var"]
        else:
            var_group = self._file["var"]

        # Try _index first (standard AnnData convention)
        if "_index" in var_group:
            names = var_group["_index"][:]
        elif "index" in var_group:
            names = var_group["index"][:]
        else:
            # Fallback: try gene_name or other columns
            for col in ("gene_name", "feature_name", "gene_symbol"):
                if col in var_group:
                    names = var_group[col][:]
                    break
            else:
                raise ValueError("Cannot find gene names in H5AD var metadata")

        return [n.decode() if isinstance(n, bytes) else str(n) for n in names]

    def read_var_df_columns(self) -> list:
        """Return available column names in the var group."""
        if self._use_raw and "raw" in self._file:
            var_group = self._file["raw/var"]
        else:
            var_group = self._file["var"]
        return list(var_group.keys())

    def read_var_column(self, col: str) -> list:
        """Read a single column from var metadata."""
        if self._use_raw and "raw" in self._file:
            var_group = self._file["raw/var"]
        else:
            var_group = self._file["var"]

        if col not in var_group:
            raise KeyError(f"Column '{col}' not found in var metadata")

        vals = var_group[col][:]
        return [v.decode() if isinstance(v, bytes) else str(v) for v in vals]

    def read_obs_names(self) -> list:
        """Read cell/observation names (barcodes)."""
        obs_group = self._file["obs"]
        if "_index" in obs_group:
            names = obs_group["_index"][:]
        elif "index" in obs_group:
            names = obs_group["index"][:]
        else:
            raise ValueError("Cannot find cell names in H5AD obs metadata")

        return [n.decode() if isinstance(n, bytes) else str(n) for n in names]

    def read_obs_column(self, col: str) -> np.ndarray:
        """Read a single column from obs metadata.

        Returns the raw array (may be categorical codes + categories).
        """
        obs_group = self._file["obs"]
        if col not in obs_group:
            raise KeyError(
                f"Column '{col}' not found in obs. "
                f"Available: {list(obs_group.keys())}"
            )

        ds = obs_group[col]
        # Handle categorical encoding
        if isinstance(ds, h5py.Group):
            # AnnData categorical: codes + categories
            codes = ds["codes"][:]
            categories = ds["categories"][:]
            categories = [c.decode() if isinstance(c, bytes) else str(c) for c in categories]
            return np.array([categories[c] if c >= 0 else "" for c in codes])
        else:
            vals = ds[:]
            if vals.dtype.kind in ("S", "O"):
                return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])
            return vals

    # -- gene name resolution -------------------------------------------------

    def resolve_gene_names(self, verbose: bool = False) -> list:
        """Resolve gene names, preferring symbols over Ensembl IDs.

        Mirrors ``_resolve_gene_names()`` from inference.py.
        """
        names = self.read_var_names()

        # Check if names look like Ensembl IDs
        n_ensembl = sum(1 for g in names if g.startswith(("ENSG", "ENSMUSG")))
        if n_ensembl > len(names) * 0.5:
            var_cols = self.read_var_df_columns()
            for col in ("gene_name", "feature_name", "gene_symbol"):
                if col in var_cols:
                    resolved = self.read_var_column(col)
                    if verbose:
                        print(f"  Resolved Ensembl IDs to gene symbols using var['{col}']")
                    return resolved

        return names

    # -- chunk iteration ------------------------------------------------------

    def iter_chunks(self):
        """Yield (start, end, csr_matrix) cell-chunks.

        Each chunk is a scipy CSR matrix of shape (chunk_cells, n_vars).
        For CSC-encoded files, chunks are read via column-to-row transpose.
        """
        if self._encoding == "csr":
            yield from self._iter_chunks_csr()
        else:
            yield from self._iter_chunks_csc()

    def _iter_chunks_csr(self):
        """Read CSR row-chunks directly from H5AD."""
        indptr_ds = self._x_group["indptr"]
        data_ds = self._x_group["data"]
        indices_ds = self._x_group["indices"]

        for start in range(0, self.n_obs, self.chunk_size):
            end = min(start + self.chunk_size, self.n_obs)

            # Read indptr slice
            indptr_slice = indptr_ds[start : end + 1]
            nnz_start = int(indptr_slice[0])
            nnz_end = int(indptr_slice[-1])

            if nnz_end > nnz_start:
                data_slice = data_ds[nnz_start:nnz_end]
                indices_slice = indices_ds[nnz_start:nnz_end]
            else:
                data_slice = np.array([], dtype=data_ds.dtype)
                indices_slice = np.array([], dtype=indices_ds.dtype)

            local_indptr = indptr_slice - nnz_start

            chunk = sps.csr_matrix(
                (data_slice, indices_slice, local_indptr),
                shape=(end - start, self.n_vars),
            )
            yield start, end, chunk

    def _iter_chunks_csc(self):
        """Read CSC-encoded files by loading column chunks and transposing.

        For CSC encoding, cells are columns. We read all columns but subset
        rows (genes) — but since we need cell-chunks, we read the full matrix
        in gene-chunks and assemble. This is less efficient than CSR, so we
        fall back to reading the full sparse matrix and slicing rows.
        """
        # For CSC, it's more efficient to read the full sparse matrix once
        # and iterate over row slices. This still avoids densification.
        indptr_ds = self._x_group["indptr"]
        data_ds = self._x_group["data"]
        indices_ds = self._x_group["indices"]

        # Read full CSC
        data = data_ds[:]
        indices = indices_ds[:]
        indptr = indptr_ds[:]

        full_csc = sps.csc_matrix(
            (data, indices, indptr), shape=(self.n_obs, self.n_vars)
        )
        del data, indices, indptr

        # Convert to CSR for efficient row slicing
        full_csr = full_csc.tocsr()
        del full_csc

        for start in range(0, self.n_obs, self.chunk_size):
            end = min(start + self.chunk_size, self.n_obs)
            yield start, end, full_csr[start:end]

    @property
    def n_chunks(self) -> int:
        return math.ceil(self.n_obs / self.chunk_size)


# =============================================================================
# Chunk Normalization
# =============================================================================


def normalize_chunk(
    chunk_csr: sps.spmatrix,
    common_gene_idx: np.ndarray,
    scale_factor: float = 1e5,
) -> sps.csc_matrix:
    """Normalize a cell-chunk and subset to common genes.

    Input:  (chunk_cells, n_genes) CSR from H5AD
    Output: (n_common_genes, chunk_cells) CSC — ready for ridge regression

    Steps:
      1. Transpose → (n_genes, chunk_cells)
      2. CPM per-cell on non-zeros
      3. log2(x + 1) on non-zeros
      4. Subset rows to common_gene_idx

    Parameters
    ----------
    chunk_csr : sparse matrix
        (chunk_cells, n_genes) chunk from H5AD.
    common_gene_idx : array-like of int
        Row indices (gene indices) to keep after transpose.
    scale_factor : float
        Library size normalization target (default 1e5 for CPM/10).
    """
    n_cells = chunk_csr.shape[0]

    # Transpose: (cells, genes) → (genes, cells) — work in CSC for col ops
    chunk_t = chunk_csr.T.tocsc()  # (n_genes, chunk_cells)

    # CPM per cell: scale each column (cell) by its total count
    col_sums = np.asarray(chunk_t.sum(axis=0)).ravel()
    # Avoid division by zero for empty cells
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    from scipy.sparse import diags as _sp_diags

    scaling = _sp_diags(scale_factor / col_sums)
    chunk_t = chunk_t.astype(np.float64) @ scaling

    # log2(x + 1) on non-zeros only (zero-preserving)
    chunk_t = chunk_t.tocsc()
    chunk_t.data = np.log2(chunk_t.data + 1)

    # Subset to common genes (row selection on CSC → convert to CSR first)
    chunk_t_csr = chunk_t.tocsr()
    Y_chunk = chunk_t_csr[common_gene_idx, :]

    return Y_chunk.tocsc()


# =============================================================================
# Streaming Statistics Accumulator
# =============================================================================


@dataclass
class _StreamingStatsAccumulator:
    """Accumulates statistics across chunks for two-pass streaming."""

    n_genes: int  # number of common genes
    n_cells: int = 0

    # Accumulate across all chunks
    row_sums: np.ndarray = field(default=None)  # (n_genes,) sum per gene across all cells
    grand_sum: float = 0.0

    # Per-cell stats (accumulated as lists, concatenated between passes)
    col_sums_list: list = field(default_factory=list)
    col_sum_sq_list: list = field(default_factory=list)

    def __post_init__(self):
        if self.row_sums is None:
            self.row_sums = np.zeros(self.n_genes, dtype=np.float64)

    def accumulate(self, Y_chunk: sps.spmatrix):
        """Accumulate stats from a (n_common_genes, chunk_cells) chunk."""
        n_g, n_c = Y_chunk.shape

        # Row sums (gene-wise, across cells)
        self.row_sums += np.asarray(Y_chunk.sum(axis=1)).ravel()

        # Column sums and squared sums (cell-wise)
        col_sums = np.asarray(Y_chunk.sum(axis=0)).ravel()
        Y_sq = Y_chunk.multiply(Y_chunk)
        col_sum_sq = np.asarray(Y_sq.sum(axis=0)).ravel()

        self.col_sums_list.append(col_sums)
        self.col_sum_sq_list.append(col_sum_sq)

        self.grand_sum += col_sums.sum()
        self.n_cells += n_c

    def finalize(self) -> tuple:
        """Compute final statistics matching _compute_population_stats(row_center=True).

        Returns
        -------
        stats : _PopulationStats
            Population statistics identical to those from the non-streaming path.
        col_sums_all : np.ndarray
            (n_cells,) column sums for all cells (concatenated chunk order).
        col_sum_sq_all : np.ndarray
            (n_cells,) column squared sums for all cells.
        """
        n = self.n_genes
        N = self.n_cells

        row_means = self.row_sums / N

        # Concatenate per-cell arrays
        col_sums_all = np.concatenate(self.col_sums_list)
        col_sum_sq_all = np.concatenate(self.col_sum_sq_list)

        # Column means of row-centered data: mu'_j = mu_j - grand_mean
        mu_c = col_sums_all / n
        grand_mean = self.grand_sum / (n * N)
        mu = mu_c - grand_mean

        # Column variance of row-centered data:
        # var(Y'_j) = (1/(n-1)) * [sum_i Y_ij^2 - 2*cross_j + sum(mu_r^2) - n*mu'_j^2]
        # We need cross_j = sum_i Y_ij * mu_r_i — this requires pass 2 data.
        # BUT: We can compute it partially. For pass 1, store what we can,
        # and defer cross computation to pass 2.
        #
        # Actually, cross = Y.T @ row_means requires the full Y. In the
        # non-streaming path, this is computed once from the full sparse Y.
        # In streaming, we need to accumulate it in pass 2.
        #
        # We return partial stats here; cross will be accumulated in pass 2.
        sum_mu_r_sq = np.sum(row_means ** 2)

        return row_means, col_sums_all, col_sum_sq_all, mu, sum_mu_r_sq, grand_mean


# =============================================================================
# Two-Pass Streaming Orchestrator
# =============================================================================


def ridge_batch_streaming(
    X: np.ndarray,
    h5ad_path: str,
    common_gene_idx: np.ndarray,
    scale_factor: float = 1e5,
    lambda_: float = DEFAULT_LAMBDA,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    chunk_size: int = 50_000,
    batch_size: int = 5000,
    backend: str = "auto",
    use_gsl_rng: bool = True,
    rng_method: Optional[str] = None,
    use_cache: bool = False,
    output_path: Optional[str] = None,
    output_compression: Optional[str] = "gzip",
    feature_names: Optional[list] = None,
    sample_names: Optional[list] = None,
    sparse_mode: bool = False,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """Two-pass streaming ridge regression on H5AD file.

    Pass 1: Read chunks, normalize, accumulate row_sums and per-cell stats.
    Pass 2: Re-read chunks, compute cross terms, run inference in sub-batches.

    Results are numerically identical to the non-streaming in-memory path.

    Parameters
    ----------
    X : ndarray, shape (n_common_genes, n_features)
        Z-scored signature matrix aligned to common genes.
    h5ad_path : str
        Path to H5AD file.
    common_gene_idx : ndarray of int
        Indices into the H5AD gene axis for common genes.
    scale_factor : float
        Library size normalization target.
    lambda_ : float
        Ridge regularization parameter.
    n_rand : int
        Number of permutations.
    seed : int
        Random seed.
    chunk_size : int
        Number of cells per H5AD read chunk.
    batch_size : int
        Number of cells per ridge regression sub-batch.
    backend : str
        "auto", "numpy", or "cupy".
    use_gsl_rng : bool
        Use GSL-compatible RNG.
    rng_method : str or None
        Explicit RNG backend.
    use_cache : bool
        Cache permutation tables.
    output_path : str or None
        If set, stream results to HDF5.
    output_compression : str or None
        Compression for output HDF5.
    feature_names : list or None
        Feature/protein names for output.
    sample_names : list or None
        Cell names for output.
    sparse_mode : bool
        Use sparse-preserving batch processing.
    verbose : bool
        Print progress.

    Returns
    -------
    dict or None
        If output_path is None, dict with 'beta', 'se', 'zscore', 'pvalue'.
        If output_path is set, None (results written to file).
    """
    start_time = time.time()
    X = np.asarray(X, dtype=np.float64)
    n_genes, n_features = X.shape

    # Backend selection
    if backend == "auto":
        backend = "cupy" if CUPY_AVAILABLE else "numpy"
    elif backend == "cupy" and not CUPY_AVAILABLE:
        raise ImportError("CuPy backend requested but not available")
    use_gpu = backend == "cupy"

    # =========================================================================
    # PASS 1: Accumulate statistics
    # =========================================================================
    if verbose:
        print("  PASS 1: Accumulating statistics...")
        t1 = time.time()

    reader = H5ADChunkReader(h5ad_path, chunk_size=chunk_size)
    n_cells_total = reader.n_obs
    accumulator = _StreamingStatsAccumulator(n_genes=n_genes)

    # Also accumulate cross = Y.T @ row_means, but we need row_means first.
    # So in pass 1 we just collect row_sums and per-cell stats.
    # We also need Y_sq_col_sums for variance computation.

    for chunk_idx, (start, end, chunk_csr) in enumerate(reader.iter_chunks()):
        Y_chunk = normalize_chunk(chunk_csr, common_gene_idx, scale_factor)
        accumulator.accumulate(Y_chunk)
        del chunk_csr, Y_chunk
        if verbose and (chunk_idx + 1) % 10 == 0:
            print(f"    Pass 1 chunk {chunk_idx + 1}/{reader.n_chunks}")

    reader.close()

    if verbose:
        print(f"  Pass 1 done: {accumulator.n_cells} cells in {time.time() - t1:.1f}s")

    # Finalize pass 1 stats
    row_means, col_sums_all, col_sum_sq_all, mu_partial, sum_mu_r_sq, grand_mean = (
        accumulator.finalize()
    )
    n_cells = accumulator.n_cells

    # =========================================================================
    # PASS 2 PREP: Precompute projection and permutation
    # =========================================================================
    if verbose:
        print("  Precomputing projection matrix T...")
    proj = _compute_projection_components(X, lambda_)

    if verbose:
        print("  Loading inverse permutation table...")
    rng_obj, use_deterministic = _get_rng(rng_method, use_gsl_rng, seed)
    if use_deterministic:
        if use_cache:
            inv_perm_table = get_cached_inverse_perm_table(n_genes, n_rand, seed, verbose=verbose)
        else:
            inv_perm_table = rng_obj.inverse_permutation_table(n_genes, n_rand)
    else:
        if verbose:
            print("  Generating permutation table (fast NumPy RNG)...")
        inv_perm_table = generate_inverse_permutation_table_fast(n_genes, n_rand, seed)

    # =========================================================================
    # PASS 2: Accumulate cross terms, then process in sub-batches
    # =========================================================================
    # We need cross_j = Y[:,j].T @ row_means for the variance computation.
    # This requires re-reading all chunks.

    if verbose:
        print("  PASS 2: Computing cross terms and running inference...")
        t2 = time.time()

    # Accumulate cross terms
    cross_all_list = []
    reader = H5ADChunkReader(h5ad_path, chunk_size=chunk_size)

    for chunk_idx, (start, end, chunk_csr) in enumerate(reader.iter_chunks()):
        Y_chunk = normalize_chunk(chunk_csr, common_gene_idx, scale_factor)
        # cross_chunk[j] = Y_chunk[:, j].T @ row_means for each cell j in chunk
        # Y_chunk is (n_genes, chunk_cells), row_means is (n_genes,)
        cross_chunk = np.asarray(Y_chunk.T @ row_means).ravel()
        cross_all_list.append(cross_chunk)
        del chunk_csr, Y_chunk
        if verbose and (chunk_idx + 1) % 10 == 0:
            print(f"    Pass 2 cross-terms chunk {chunk_idx + 1}/{reader.n_chunks}")

    reader.close()

    cross_all = np.concatenate(cross_all_list)
    del cross_all_list

    # Now compute full population stats matching _compute_population_stats(row_center=True)
    # mu'_j = col_sums_j / n_genes - grand_mean  (already computed as mu_partial)
    mu = mu_partial

    # Variance of row-centered data:
    # var(Y'_j) = (1/(n-1)) * [Y_sq_col_sums_j - 2*cross_j + sum(mu_r^2) - n*mu_j'^2]
    numerator = col_sum_sq_all - 2 * cross_all + sum_mu_r_sq - n_genes * mu ** 2
    variance = numerator / (n_genes - 1)
    variance = np.maximum(variance, 0)

    sigma = np.sqrt(variance)
    sigma = np.where(sigma < EPS, 1.0, sigma)
    mu_over_sigma = mu / sigma

    full_stats = _PopulationStats(
        mu=mu,
        sigma=sigma,
        mu_over_sigma=mu_over_sigma,
        n_genes=n_genes,
        row_means=row_means,
    )

    del col_sums_all, col_sum_sq_all, cross_all, mu_partial
    del variance, numerator
    gc.collect()

    if verbose:
        print(f"  Population stats computed. Now processing inference batches...")

    # =========================================================================
    # PASS 3: Re-read chunks and process inference sub-batches
    # =========================================================================
    # We accumulate cells across chunks, then feed sub-batches of size
    # `batch_size` to the existing batch processing functions.

    # Setup streaming output
    writer = None
    if output_path is not None:
        if verbose:
            print(f"  Output: streaming to {output_path}")
        writer = StreamingResultWriter(
            output_path,
            n_features=n_features,
            n_samples=n_cells,
            feature_names=feature_names,
            sample_names=sample_names,
            compression=output_compression,
        )

    results_list = [] if writer is None else None
    cells_processed = 0
    n_batches_done = 0

    reader = H5ADChunkReader(h5ad_path, chunk_size=chunk_size)

    # Buffer for accumulating cells from chunks into sub-batches
    buffer_chunks = []
    buffer_n_cells = 0

    def _flush_buffer(buf_chunks, global_col_start):
        """Process all cells in buffer as sub-batches."""
        nonlocal n_batches_done

        if not buf_chunks:
            return global_col_start

        # Concatenate buffered chunks horizontally
        if len(buf_chunks) == 1:
            Y_buf = buf_chunks[0]
        else:
            Y_buf = sps.hstack(buf_chunks, format="csc")

        buf_n = Y_buf.shape[1]

        # Process in sub-batches
        for sub_start in range(0, buf_n, batch_size):
            sub_end = min(sub_start + batch_size, buf_n)
            Y_sub = Y_buf[:, sub_start:sub_end]

            # Slice stats for this sub-batch
            abs_col_start = global_col_start + sub_start
            abs_col_end = global_col_start + sub_end
            batch_stats = full_stats.slice(abs_col_start, abs_col_end)

            if use_gpu:
                batch_result = _process_sparse_batch_cupy(
                    proj.T, proj.c, Y_sub,
                    batch_stats.sigma, batch_stats.mu_over_sigma,
                    inv_perm_table, n_rand,
                    sparse_mode=sparse_mode,
                    row_means=batch_stats.row_means,
                    col_center=True,
                    col_scale=True,
                    mu=batch_stats.mu,
                )
            else:
                batch_result = _process_sparse_batch_numpy(
                    proj.T, proj.c, Y_sub,
                    batch_stats.sigma, batch_stats.mu_over_sigma,
                    inv_perm_table, n_rand,
                    sparse_mode=sparse_mode,
                    row_means=batch_stats.row_means,
                    col_center=True,
                    col_scale=True,
                    mu=batch_stats.mu,
                )

            if writer is not None:
                writer.write_batch(batch_result, start_col=abs_col_start)
            else:
                results_list.append(batch_result)

            n_batches_done += 1
            del Y_sub, batch_stats, batch_result

        new_start = global_col_start + buf_n
        del Y_buf
        gc.collect()
        return new_start

    # Main loop: read chunks and buffer them
    global_col = 0
    for chunk_idx, (start, end, chunk_csr) in enumerate(reader.iter_chunks()):
        Y_chunk = normalize_chunk(chunk_csr, common_gene_idx, scale_factor)
        del chunk_csr

        chunk_n = Y_chunk.shape[1]
        buffer_chunks.append(Y_chunk)
        buffer_n_cells += chunk_n

        # Flush when buffer is large enough
        if buffer_n_cells >= batch_size:
            global_col = _flush_buffer(buffer_chunks, global_col)
            cells_processed += buffer_n_cells
            buffer_chunks = []
            buffer_n_cells = 0

            if verbose:
                elapsed = time.time() - t2
                rate = cells_processed / elapsed if elapsed > 0 else 0
                print(
                    f"    {cells_processed}/{n_cells} cells, "
                    f"{n_batches_done} batches, "
                    f"{rate:.0f} cells/s"
                )

    # Flush remaining buffer
    if buffer_chunks:
        global_col = _flush_buffer(buffer_chunks, global_col)
        cells_processed += buffer_n_cells
        buffer_chunks = []

    reader.close()

    if verbose:
        print(
            f"  Pass 2+3 done: {cells_processed} cells, "
            f"{n_batches_done} batches in {time.time() - t2:.1f}s"
        )

    # Finalize
    total_time = time.time() - start_time
    del proj, full_stats, inv_perm_table
    if use_gpu and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    if writer is not None:
        writer.close()
        if verbose:
            print(f"  Results written to {output_path}")
            print(f"  Completed in {total_time:.1f}s")
        return None

    # Concatenate in-memory results
    if verbose:
        print("  Concatenating results...")

    final_result = {
        "beta": np.hstack([r["beta"] for r in results_list]),
        "se": np.hstack([r["se"] for r in results_list]),
        "zscore": np.hstack([r["zscore"] for r in results_list]),
        "pvalue": np.hstack([r["pvalue"] for r in results_list]),
        "method": f"{backend}_streaming",
        "time": total_time,
        "n_batches": n_batches_done,
    }

    if verbose:
        print(f"  Completed in {total_time:.1f}s")

    return final_result


# =============================================================================
# High-Level Wrappers
# =============================================================================


def _resolve_and_prepare_genes(
    reader: H5ADChunkReader,
    sig_matrix,
    is_group_sig: bool,
    is_group_cor: float,
    sig_filter: bool,
    sort_genes: bool,
    verbose: bool,
) -> tuple:
    """Resolve gene names from H5AD and prepare signature.

    Returns
    -------
    gene_names : list
        Standardized gene names from H5AD.
    X_scaled : DataFrame
        Z-scored signature aligned to common genes.
    common_genes : list
        Common genes between expression and signature.
    common_gene_idx : ndarray
        Indices into gene_names for common genes.
    """
    import pandas as pd

    from .inference import _prepare_signature_for_sparse

    # Resolve gene names
    gene_names = reader.resolve_gene_names(verbose=verbose)

    # Standardize (uppercase, remove version numbers)
    gene_names = [g.upper() for g in gene_names]
    gene_names = [g.split(".")[0] if "." in g else g for g in gene_names]

    # Handle duplicates: need to estimate gene means from a small sample
    if len(gene_names) != len(set(gene_names)):
        if verbose:
            print("  Estimating gene means for deduplication (first 1000 cells)...")

        # Read a small sample to estimate means
        n_sample = min(1000, reader.n_obs)
        sample_reader = H5ADChunkReader(reader.path, chunk_size=n_sample)
        _, _, sample_chunk = next(sample_reader.iter_chunks())
        sample_reader.close()

        # Compute mean per gene from sample
        if sps.issparse(sample_chunk):
            gene_means = np.asarray(sample_chunk.mean(axis=0)).ravel()
        else:
            gene_means = np.mean(sample_chunk, axis=0)

        gene_to_best_idx = {}
        for idx, gene in enumerate(gene_names):
            if gene not in gene_to_best_idx or gene_means[idx] > gene_means[gene_to_best_idx[gene]]:
                gene_to_best_idx[gene] = idx

        keep_idx = sorted(gene_to_best_idx.values())
        # Build mapping from original index to deduplicated index
        dedup_gene_names = [gene_names[i] for i in keep_idx]

        if verbose:
            print(f"  Deduplication: {len(gene_names)} → {len(dedup_gene_names)} genes")

        gene_names = dedup_gene_names
        # keep_idx maps new positions to original H5AD gene indices
        orig_gene_idx = np.array(keep_idx)
    else:
        orig_gene_idx = None  # No dedup needed — identity mapping

    # Prepare signature
    X_scaled, common_genes = _prepare_signature_for_sparse(
        gene_names=gene_names,
        sig_matrix=sig_matrix,
        is_group_sig=is_group_sig,
        is_group_cor=is_group_cor,
        sig_filter=sig_filter,
        sort_genes=sort_genes,
        verbose=verbose,
    )

    # Build common gene index (into original H5AD gene axis)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    common_local_idx = [gene_to_idx[g] for g in common_genes]

    if orig_gene_idx is not None:
        # Map through dedup: local idx → original H5AD idx
        common_gene_idx = np.array([orig_gene_idx[i] for i in common_local_idx])
    else:
        common_gene_idx = np.array(common_local_idx)

    return gene_names, X_scaled, common_genes, common_gene_idx


def run_streaming_scrnaseq(
    h5ad_path: str,
    cell_type_col: str,
    sig_matrix="secact",
    is_group_sig: bool = True,
    is_group_cor: float = 0.9,
    lambda_: float = DEFAULT_LAMBDA,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    sig_filter: bool = False,
    backend: str = "auto",
    use_gsl_rng: bool = True,
    rng_method: Optional[str] = None,
    use_cache: bool = False,
    batch_size: int = 5000,
    output_path: Optional[str] = None,
    output_compression: Optional[str] = "gzip",
    sort_genes: bool = False,
    sparse_mode: bool = False,
    chunk_size: int = 50_000,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """Streaming single-cell inference from H5AD file.

    Called by ``secact_activity_inference_scrnaseq(streaming=True)``.
    """
    import pandas as pd

    from .inference import expand_rows

    if verbose:
        print("SecActPy Streaming scRNAseq Activity Inference")
        print("=" * 50)
        print(f"  File: {h5ad_path}")

    # Open reader and resolve genes
    reader = H5ADChunkReader(h5ad_path, chunk_size=chunk_size)

    if verbose:
        print(f"  H5AD shape: {reader.n_obs} cells × {reader.n_vars} genes")
        print(f"  Chunk size: {chunk_size}")

    # Validate cell_type_col exists
    try:
        cell_types = reader.read_obs_column(cell_type_col)
    except KeyError as e:
        reader.close()
        raise ValueError(str(e))

    cell_names = reader.read_obs_names()

    if verbose:
        unique_types = set(cell_types)
        print(f"  Cell types ({len(unique_types)}): {list(unique_types)[:5]}...")

    # Resolve genes and prepare signature
    gene_names, X_scaled, common_genes, common_gene_idx = _resolve_and_prepare_genes(
        reader=reader,
        sig_matrix=sig_matrix,
        is_group_sig=is_group_sig,
        is_group_cor=is_group_cor,
        sig_filter=sig_filter,
        sort_genes=sort_genes,
        verbose=verbose,
    )
    reader.close()

    if verbose:
        print(f"  Common genes: {len(common_genes)}")
        print(f"  Signature: {X_scaled.shape[0]} genes × {X_scaled.shape[1]} features")

    # Run streaming ridge batch
    result = ridge_batch_streaming(
        X=X_scaled.values,
        h5ad_path=h5ad_path,
        common_gene_idx=common_gene_idx,
        scale_factor=1e5,  # CPM/10 for scRNAseq
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        chunk_size=chunk_size,
        batch_size=batch_size,
        backend=backend,
        use_gsl_rng=use_gsl_rng,
        rng_method=rng_method,
        use_cache=use_cache,
        output_path=output_path,
        output_compression=output_compression,
        feature_names=X_scaled.columns.tolist(),
        sample_names=cell_names,
        sparse_mode=sparse_mode,
        verbose=verbose,
    )

    if result is None:
        if verbose:
            print(f"  Results streamed to {output_path}")
        return None

    # Format results as DataFrames
    feature_names = X_scaled.columns.tolist()
    beta_df = pd.DataFrame(result["beta"], index=feature_names, columns=cell_names)
    se_df = pd.DataFrame(result["se"], index=feature_names, columns=cell_names)
    zscore_df = pd.DataFrame(result["zscore"], index=feature_names, columns=cell_names)
    pvalue_df = pd.DataFrame(result["pvalue"], index=feature_names, columns=cell_names)

    if is_group_sig:
        if verbose:
            print("  Expanding grouped signatures...")
        beta_df = expand_rows(beta_df)
        se_df = expand_rows(se_df)
        zscore_df = expand_rows(zscore_df)
        pvalue_df = expand_rows(pvalue_df)

        row_order = sorted(beta_df.index)
        beta_df = beta_df.loc[row_order]
        se_df = se_df.loc[row_order]
        zscore_df = zscore_df.loc[row_order]
        pvalue_df = pvalue_df.loc[row_order]

    if verbose:
        print(f"  Result shape: {beta_df.shape}")

    return {
        "beta": beta_df,
        "se": se_df,
        "zscore": zscore_df,
        "pvalue": pvalue_df,
    }


def run_streaming_st(
    h5ad_path: str,
    sig_matrix="secact",
    is_group_sig: bool = True,
    is_group_cor: float = 0.9,
    scale_factor: float = 1e5,
    lambda_: float = DEFAULT_LAMBDA,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    sig_filter: bool = False,
    backend: str = "auto",
    use_gsl_rng: bool = True,
    rng_method: Optional[str] = None,
    use_cache: bool = False,
    batch_size: int = 5000,
    output_path: Optional[str] = None,
    output_compression: Optional[str] = "gzip",
    sort_genes: bool = False,
    sparse_mode: bool = False,
    chunk_size: int = 50_000,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """Streaming spatial transcriptomics inference from H5AD file.

    Called by ``secact_activity_inference_st(streaming=True)``.
    """
    import pandas as pd

    from .inference import expand_rows

    if verbose:
        print("SecActPy Streaming ST Activity Inference")
        print("=" * 50)
        print(f"  File: {h5ad_path}")

    reader = H5ADChunkReader(h5ad_path, chunk_size=chunk_size)

    if verbose:
        print(f"  H5AD shape: {reader.n_obs} spots × {reader.n_vars} genes")
        print(f"  Chunk size: {chunk_size}")

    spot_names = reader.read_obs_names()

    # Resolve genes and prepare signature
    gene_names, X_scaled, common_genes, common_gene_idx = _resolve_and_prepare_genes(
        reader=reader,
        sig_matrix=sig_matrix,
        is_group_sig=is_group_sig,
        is_group_cor=is_group_cor,
        sig_filter=sig_filter,
        sort_genes=sort_genes,
        verbose=verbose,
    )
    reader.close()

    if verbose:
        print(f"  Common genes: {len(common_genes)}")
        print(f"  Signature: {X_scaled.shape[0]} genes × {X_scaled.shape[1]} features")

    # Run streaming ridge batch
    result = ridge_batch_streaming(
        X=X_scaled.values,
        h5ad_path=h5ad_path,
        common_gene_idx=common_gene_idx,
        scale_factor=scale_factor,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        chunk_size=chunk_size,
        batch_size=batch_size,
        backend=backend,
        use_gsl_rng=use_gsl_rng,
        rng_method=rng_method,
        use_cache=use_cache,
        output_path=output_path,
        output_compression=output_compression,
        feature_names=X_scaled.columns.tolist(),
        sample_names=spot_names,
        sparse_mode=sparse_mode,
        verbose=verbose,
    )

    if result is None:
        if verbose:
            print(f"  Results streamed to {output_path}")
        return None

    # Format results as DataFrames
    feature_names = X_scaled.columns.tolist()
    beta_df = pd.DataFrame(result["beta"], index=feature_names, columns=spot_names)
    se_df = pd.DataFrame(result["se"], index=feature_names, columns=spot_names)
    zscore_df = pd.DataFrame(result["zscore"], index=feature_names, columns=spot_names)
    pvalue_df = pd.DataFrame(result["pvalue"], index=feature_names, columns=spot_names)

    if is_group_sig:
        if verbose:
            print("  Expanding grouped signatures...")
        beta_df = expand_rows(beta_df)
        se_df = expand_rows(se_df)
        zscore_df = expand_rows(zscore_df)
        pvalue_df = expand_rows(pvalue_df)

        row_order = sorted(beta_df.index)
        beta_df = beta_df.loc[row_order]
        se_df = se_df.loc[row_order]
        zscore_df = zscore_df.loc[row_order]
        pvalue_df = pvalue_df.loc[row_order]

    if verbose:
        print(f"  Result shape: {beta_df.shape}")

    return {
        "beta": beta_df,
        "se": se_df,
        "zscore": zscore_df,
        "pvalue": pvalue_df,
    }
