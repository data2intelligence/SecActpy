#!/usr/bin/env python3
"""
SecActPy Command Line Interface

Secreted protein activity inference from gene expression data.

Usage:
    secactpy bulk -i <input> -o <output> [options]
    secactpy scrnaseq -i <input> -o <output> [options]
    secactpy visium -i <input> -o <output> [options]
    secactpy cosmx -i <input> -o <output> [options]

Examples:
    # Bulk RNA-seq (differential expression)
    secactpy bulk -i diff_expr.tsv -o results.h5ad --differential

    # scRNA-seq (h5ad or 10x)
    secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype

    # Visium spatial transcriptomics
    secactpy visium -i /path/to/visium/ -o results.h5ad

    # CosMx spatial transcriptomics
    secactpy cosmx -i data.h5ad -o results.h5ad --cell-type-col cell_type
"""

import argparse
import sys
from pathlib import Path


# Detect CuPy/GPU availability at import time
def _detect_gpu():
    """Detect if CuPy and GPU are available."""
    try:
        import cupy as cp
        # Try to actually use the GPU
        cp.array([1, 2, 3])
        return True
    except Exception:
        return False

CUPY_AVAILABLE = _detect_gpu()
DEFAULT_BACKEND = "cupy" if CUPY_AVAILABLE else "numpy"


def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output H5AD file"
    )
    parser.add_argument(
        "-s", "--signature",
        default="secact",
        choices=["secact", "cytosig"],
        help="Signature matrix (default: secact)"
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=5e5,
        help="Ridge regularization parameter (default: 5e5)"
    )
    parser.add_argument(
        "-n", "--n-rand",
        type=int,
        default=1000,
        help="Number of permutations (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "cupy"],
        default=DEFAULT_BACKEND,
        help=f"Computation backend (default: {DEFAULT_BACKEND})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for large datasets (default: auto)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Cache permutation tables to disk"
    )
    parser.add_argument(
        "--sort-genes",
        action="store_true",
        help="Sort genes alphabetically before ridge regression (NOT recommended for R compatibility)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output"
    )


def cmd_bulk(args: argparse.Namespace) -> int:
    """Run bulk RNA-seq inference."""
    from secactpy import secact_activity_inference
    from secactpy.io import save_results_to_h5ad
    import pandas as pd

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - Bulk RNA-seq Inference")
        print("=" * 60)
        print(f"Input:     {args.input}")
        print(f"Output:    {args.output}")
        print(f"Signature: {args.signature}")
        print(f"Lambda:    {args.lambda_}")
        print(f"N_rand:    {args.n_rand}")
        print(f"Backend:   {args.backend}")
        print("=" * 60)

    # Load input
    input_path = Path(args.input)
    if input_path.suffix in [".csv"]:
        expr = pd.read_csv(input_path, index_col=0)
    elif input_path.suffix in [".tsv", ".txt"]:
        # Try tab-separated first
        expr = pd.read_csv(input_path, sep="\t", index_col=0)

        # If no columns (0 samples), try space-separated format
        # This handles files like: "Diff\nGENE1 0.5\nGENE2 -0.3\n..."
        if expr.shape[1] == 0:
            if verbose:
                print("Tab-separated parsing failed. Trying space-separated format...")
            expr = pd.read_csv(input_path, sep=r"\s+", index_col=0)

        # Still no columns? Try without index_col (some formats have gene names as a column)
        if expr.shape[1] == 0:
            if verbose:
                print("Trying alternative parsing without index_col...")
            expr = pd.read_csv(input_path, sep=r"\s+")
            # Use first column as index
            if expr.shape[1] >= 1:
                first_col = expr.columns[0]
                expr = expr.set_index(first_col)

    elif input_path.suffix in [".h5ad"]:
        import anndata
        adata = anndata.read_h5ad(input_path)

        # Check if file needs transposition
        # Standard: obs=samples, var=genes -> to_df() gives (samples × genes) -> .T gives (genes × samples)
        # Reversed: obs=genes, var=samples -> to_df() gives (genes × samples) -> .T gives (samples × genes)

        n_obs = adata.n_obs
        n_var = adata.n_vars

        # Auto-detect: if obs >> var, likely obs=genes (reversed format)
        # Or check if --transpose flag is set
        transpose_input = getattr(args, 'transpose', False)

        if not transpose_input and n_obs > n_var and n_obs > 5000:
            # Likely reversed format (obs=genes, var=samples)
            if verbose:
                print(f"  Auto-detected reversed format: obs={n_obs} (genes), var={n_var} (samples)")
                print("  Using adata.to_df() directly (no transpose)")
            expr = adata.to_df()  # (genes × samples) - already correct
        elif transpose_input:
            if verbose:
                print("  --transpose flag set: using adata.to_df() directly")
            expr = adata.to_df()  # User requested transpose
        else:
            # Standard format (obs=samples, var=genes)
            expr = adata.to_df().T  # (samples × genes) -> (genes × samples)
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}", file=sys.stderr)
        return 1

    if verbose:
        print(f"Loaded expression: {expr.shape[0]} genes × {expr.shape[1]} samples")

    # Run inference
    result = secact_activity_inference(
        expr,
        is_differential=args.differential,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        sort_genes=args.sort_genes,
        verbose=verbose
    )

    # Save output
    save_results_to_h5ad(
        result,
        args.output,
        sample_names=list(expr.columns),
        verbose=verbose
    )

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_scrnaseq(args: argparse.Namespace) -> int:
    """Run scRNA-seq inference."""
    from secactpy import secact_activity_inference_scrnaseq

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - scRNA-seq Inference")
        print("=" * 60)
        print(f"Input:        {args.input}")
        print(f"Output:       {args.output}")
        print(f"Cell type:    {args.cell_type_col or 'None (all cells)'}")
        print(f"Single cell:  {args.single_cell}")
        print(f"Signature:    {args.signature}")
        print(f"Backend:      {args.backend}")
        print("=" * 60)

    # Run inference
    result = secact_activity_inference_scrnaseq(
        args.input,
        cell_type_col=args.cell_type_col,
        is_single_cell_level=args.single_cell,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        sort_genes=args.sort_genes,
        verbose=verbose
    )

    # Save output
    from secactpy.io import save_results_to_h5ad
    save_results_to_h5ad(result, args.output, verbose=verbose)

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_visium(args: argparse.Namespace) -> int:
    """Run Visium spatial transcriptomics inference."""
    from secactpy import secact_activity_inference_st

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - Visium Spatial Transcriptomics Inference")
        print("=" * 60)
        print(f"Input:       {args.input}")
        print(f"Output:      {args.output}")
        print(f"Cell type:   {args.cell_type_col or 'None (spot-level)'}")
        print(f"Spot level:  {args.spot_level}")
        print(f"Signature:   {args.signature}")
        print(f"Backend:     {args.backend}")
        print("=" * 60)

    # Run inference
    result = secact_activity_inference_st(
        args.input,
        cell_type_col=args.cell_type_col,
        is_spot_level=args.spot_level,
        min_genes=args.min_genes,
        scale_factor=args.scale_factor,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        sort_genes=args.sort_genes,
        verbose=verbose
    )

    # Save output
    from secactpy.io import save_results_to_h5ad
    save_results_to_h5ad(result, args.output, verbose=verbose)

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_cosmx(args: argparse.Namespace) -> int:
    """Run CosMx spatial transcriptomics inference."""
    from secactpy import secact_activity_inference_st

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - CosMx Spatial Transcriptomics Inference")
        print("=" * 60)
        print(f"Input:        {args.input}")
        print(f"Output:       {args.output}")
        print(f"Cell type:    {args.cell_type_col or 'None (cell-level)'}")
        print(f"Signature:    {args.signature}")
        print(f"Scale factor: {args.scale_factor}")
        print(f"Min genes:    {args.min_genes}")
        print(f"Sig filter:   {args.sig_filter}")
        print(f"Backend:      {args.backend}")
        print(f"Batch size:   {args.batch_size or 'auto'}")
        print("=" * 60)

    # Run inference (CosMx uses same ST function with different defaults)
    result = secact_activity_inference_st(
        args.input,
        cell_type_col=args.cell_type_col,
        is_spot_level=True,  # CosMx processes at cell level
        min_genes=args.min_genes,
        scale_factor=args.scale_factor,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        sort_genes=args.sort_genes,
        verbose=verbose
    )

    # Save output
    from secactpy.io import save_results_to_h5ad
    save_results_to_h5ad(result, args.output, verbose=verbose)

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two H5AD files for identity validation."""
    import numpy as np

    print("=" * 60)
    print("SecActPy - H5AD Comparison")
    print("=" * 60)
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Sort indices: {args.sort}")
    print(f"By position: {getattr(args, 'by_position', False)}")

    # Try to load files - handle R transposed format
    try:
        import h5py
    except ImportError:
        print("Error: h5py is required. Install with: pip install h5py")
        return 1

    def load_h5ad_flexible(filepath):
        """Load H5AD file, handling both R and Python formats, dense and sparse."""
        result = {}

        with h5py.File(filepath, 'r') as f:
            # Try to get X matrix - handle both dense and sparse formats
            if 'X' in f:
                x_item = f['X']
                if isinstance(x_item, h5py.Dataset):
                    # Dense matrix
                    result['X'] = x_item[:]
                elif isinstance(x_item, h5py.Group):
                    # Sparse matrix (CSR/CSC format from MuDataSeurat)
                    # Check encoding type
                    encoding = x_item.attrs.get('encoding-type', b'')
                    if isinstance(encoding, bytes):
                        encoding = encoding.decode('utf-8')

                    if 'data' in x_item and 'indices' in x_item and 'indptr' in x_item:
                        from scipy import sparse
                        data = x_item['data'][:]
                        indices = x_item['indices'][:]
                        indptr = x_item['indptr'][:]
                        shape = tuple(x_item.attrs.get('shape', []))

                        if 'csr' in encoding.lower() or shape[0] == len(indptr) - 1:
                            # CSR format
                            sp_mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
                        else:
                            # CSC format
                            sp_mat = sparse.csc_matrix((data, indices, indptr), shape=shape)

                        result['X'] = sp_mat.toarray()
                    else:
                        print("  Warning: Unknown sparse format in X group")
                        result['X'] = None

            # Get obs/var names - handle different formats
            def get_index(group_name):
                if group_name not in f:
                    return []
                grp = f[group_name]

                # Try _index first
                if '_index' in grp:
                    idx = grp['_index'][:]
                # Try index attribute
                elif 'index' in grp.attrs:
                    idx_name = grp.attrs['index']
                    if isinstance(idx_name, bytes):
                        idx_name = idx_name.decode('utf-8')
                    if idx_name in grp:
                        idx = grp[idx_name][:]
                    else:
                        idx = []
                else:
                    # Look for any dataset that might be the index
                    idx = []
                    for key in grp.keys():
                        if key.startswith('_') or key == 'index':
                            idx = grp[key][:]
                            break

                if len(idx) > 0:
                    if isinstance(idx[0], bytes):
                        idx = [x.decode('utf-8') for x in idx]
                    return list(idx)
                return []

            result['obs_names'] = get_index('obs')
            result['var_names'] = get_index('var')

            # Get obsm matrices (se, zscore, pvalue) - handle sparse
            if 'obsm' in f:
                for key in f['obsm'].keys():
                    obsm_item = f['obsm/' + key]
                    if isinstance(obsm_item, h5py.Dataset):
                        result['obsm_' + key] = obsm_item[:]
                    elif isinstance(obsm_item, h5py.Group):
                        # Sparse obsm
                        if 'data' in obsm_item and 'indices' in obsm_item and 'indptr' in obsm_item:
                            from scipy import sparse
                            data = obsm_item['data'][:]
                            indices = obsm_item['indices'][:]
                            indptr = obsm_item['indptr'][:]
                            shape = tuple(obsm_item.attrs.get('shape', []))
                            sp_mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
                            result['obsm_' + key] = sp_mat.toarray()

            # Check for reductions (Seurat/MuDataSeurat format)
            if 'reductions' in f:
                for key in f['reductions'].keys():
                    red_path = 'reductions/' + key
                    # Check for cell.embeddings within each reduction
                    if 'cell.embeddings' in f[red_path]:
                        emb_item = f[red_path + '/cell.embeddings']
                        if isinstance(emb_item, h5py.Dataset):
                            result['obsm_' + key] = emb_item[:]
                        elif isinstance(emb_item, h5py.Group):
                            if 'data' in emb_item and 'indices' in emb_item and 'indptr' in emb_item:
                                from scipy import sparse
                                data = emb_item['data'][:]
                                indices = emb_item['indices'][:]
                                indptr = emb_item['indptr'][:]
                                shape = tuple(emb_item.attrs.get('shape', []))
                                sp_mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
                                result['obsm_' + key] = sp_mat.toarray()

            # Check for layers
            if 'layers' in f:
                for key in f['layers'].keys():
                    layer_item = f['layers/' + key]
                    if isinstance(layer_item, h5py.Dataset):
                        result['layers_' + key] = layer_item[:]
                    elif isinstance(layer_item, h5py.Group):
                        if 'data' in layer_item and 'indices' in layer_item and 'indptr' in layer_item:
                            from scipy import sparse
                            data = layer_item['data'][:]
                            indices = layer_item['indices'][:]
                            indptr = layer_item['indptr'][:]
                            shape = tuple(layer_item.attrs.get('shape', []))
                            sp_mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
                            result['layers_' + key] = sp_mat.toarray()

        return result

    # Load both files
    try:
        data1 = load_h5ad_flexible(args.file1)
        n_obs1 = len(data1.get('obs_names', []))
        n_var1 = len(data1.get('var_names', []))
        x_shape1 = data1.get('X', np.array([])).shape
        print(f"\nFile 1: obs={n_obs1}, var={n_var1}, X.shape={x_shape1}")

        # Show all keys found (for debugging)
        obsm_keys1 = [k.replace('obsm_', '') for k in data1.keys() if k.startswith('obsm_')]
        layers_keys1 = [k.replace('layers_', '') for k in data1.keys() if k.startswith('layers_')]
        if obsm_keys1:
            print(f"  obsm keys: {obsm_keys1}")
        if layers_keys1:
            print(f"  layers keys: {layers_keys1}")

        # If verbose, show raw h5 structure
        if args.verbose:
            import h5py
            with h5py.File(args.file1, 'r') as f:
                def print_h5_structure(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"    [G] {name}/")
                    else:
                        print(f"    [D] {name}: {obj.shape if hasattr(obj, 'shape') else '?'}")
                print("  H5 structure:")
                f.visititems(print_h5_structure)
    except Exception as e:
        print(f"Error loading file 1: {e}")
        import traceback
        traceback.print_exc()
        return 1

    try:
        data2 = load_h5ad_flexible(args.file2)
        n_obs2 = len(data2.get('obs_names', []))
        n_var2 = len(data2.get('var_names', []))
        x_shape2 = data2.get('X', np.array([])).shape
        print(f"File 2: obs={n_obs2}, var={n_var2}, X.shape={x_shape2}")

        # Show all keys found
        obsm_keys2 = [k.replace('obsm_', '') for k in data2.keys() if k.startswith('obsm_')]
        layers_keys2 = [k.replace('layers_', '') for k in data2.keys() if k.startswith('layers_')]
        if obsm_keys2:
            print(f"  obsm keys: {obsm_keys2}")
        if layers_keys2:
            print(f"  layers keys: {layers_keys2}")
    except Exception as e:
        print(f"Error loading file 2: {e}")
        return 1

    # Detect if files need transposition alignment
    # Check if one file is transposed relative to the other
    r_format1 = False
    r_format2 = False

    # R format: obs=proteins (few), var=cells (many), X=(var×obs)=(cells×proteins)
    # The X data is already in correct shape, just metadata is swapped
    # Python format: obs=cells (many), var=proteins (few), X=(obs×var)=(cells×proteins)
    if n_obs1 < 2000 and n_var1 > 10000:
        print("  File 1: R format (obs=proteins, var=cells)")
        r_format1 = True
    if n_obs2 < 2000 and n_var2 > 10000:
        print("  File 2: R format (obs=proteins, var=cells)")
        r_format2 = True

    # Extract and align matrices
    def get_aligned_matrix(data, name, is_r_format):
        """Get matrix with consistent orientation (cells × proteins).

        For R format: obs=proteins, var=cells, X=(var×obs)=(cells×proteins)
        For Python format: obs=cells, var=proteins, X=(obs×var)=(cells×proteins)

        Both have X in same orientation, just different metadata interpretation.
        """
        # Define possible names for each matrix type
        name_variants = {
            'beta': ['X', 'beta'],
            'se': ['se', 'SE', 'SE_', 'X_se'],
            'zscore': ['zscore', 'Z', 'Z_', 'X_zscore'],
            'pvalue': ['pvalue', 'P', 'P_', 'X_pvalue']
        }

        mat = None

        if name == 'beta' or name == 'X':
            mat = data.get('X')
        else:
            # Try different possible names
            variants = name_variants.get(name, [name])
            for variant in variants:
                if 'obsm_' + variant in data:
                    mat = data['obsm_' + variant]
                    break
                elif 'layers_' + variant in data:
                    mat = data['layers_' + variant]
                    break

        if mat is None:
            return None, None, None

        if is_r_format:
            # R format: X is (var × obs) = (cells × proteins)
            # Don't transpose, just swap name interpretation
            row_names = data.get('var_names', [])   # cells (from var)
            col_names = data.get('obs_names', [])   # proteins (from obs)
        else:
            # Python format: X is (obs × var) = (cells × proteins)
            row_names = data.get('obs_names', [])   # cells
            col_names = data.get('var_names', [])   # proteins

        return mat, row_names, col_names

    # Compare matrices
    matrices_to_compare = ['beta', 'se', 'zscore', 'pvalue']
    all_identical = True

    for name in matrices_to_compare:
        mat1, rows1, cols1 = get_aligned_matrix(data1, name, r_format1)
        mat2, rows2, cols2 = get_aligned_matrix(data2, name, r_format2)

        if mat1 is None and mat2 is None:
            print(f"\n{name}: Not found in either file (skipping)")
            continue
        elif mat1 is None:
            print(f"\n❌ {name}: Only in file 2")
            all_identical = False
            continue
        elif mat2 is None:
            print(f"\n❌ {name}: Only in file 1")
            all_identical = False
            continue

        mat1 = np.asarray(mat1, dtype=np.float64)
        mat2 = np.asarray(mat2, dtype=np.float64)

        by_position = getattr(args, 'by_position', False)

        # If --by-position, compare by position (ignoring names)
        # But if proteins are same set, align them first
        if by_position:
            # Check if shapes match
            if mat1.shape != mat2.shape:
                print(f"\n❌ {name}: Shape mismatch {mat1.shape} vs {mat2.shape}")
                all_identical = False
                continue

            # If proteins are same set but different order, align by protein names only
            if len(cols1) > 0 and len(cols2) > 0 and set(cols1) == set(cols2) and cols1 != cols2:
                # Sort columns (proteins) alphabetically in both
                col_order1 = np.argsort(cols1)
                col_order2 = np.argsort(cols2)
                mat1 = mat1[:, col_order1]
                mat2 = mat2[:, col_order2]
                print(f"\n{name} (aligned proteins by name, cells by position):")
            else:
                print(f"\n{name} (by position):")
        # If --sort, align by indices
        elif args.sort and len(rows1) > 0 and len(rows2) > 0 and len(cols1) > 0 and len(cols2) > 0:
            common_rows = sorted(set(rows1) & set(rows2))
            common_cols = sorted(set(cols1) & set(cols2))

            if len(common_rows) == 0 or len(common_cols) == 0:
                print(f"\n❌ {name}: No common rows/cols after sorting")
                all_identical = False
                continue

            # Create index maps
            row_idx1 = {r: i for i, r in enumerate(rows1)}
            row_idx2 = {r: i for i, r in enumerate(rows2)}
            col_idx1 = {c: i for i, c in enumerate(cols1)}
            col_idx2 = {c: i for i, c in enumerate(cols2)}

            # Extract aligned submatrices
            ridx1 = [row_idx1[r] for r in common_rows]
            ridx2 = [row_idx2[r] for r in common_rows]
            cidx1 = [col_idx1[c] for c in common_cols]
            cidx2 = [col_idx2[c] for c in common_cols]

            mat1 = mat1[np.ix_(ridx1, cidx1)]
            mat2 = mat2[np.ix_(ridx2, cidx2)]

            print(f"\n{name} (aligned to {len(common_rows)}×{len(common_cols)}):")
        else:
            if mat1.shape != mat2.shape:
                print(f"\n❌ {name}: Shape mismatch {mat1.shape} vs {mat2.shape}")
                all_identical = False
                continue
            print(f"\n{name}:")

        # Compute differences
        diff = mat1 - mat2
        abs_diff = np.abs(diff)

        max_diff = np.nanmax(abs_diff)
        mean_diff = np.nanmean(abs_diff)

        # Correlation
        valid = ~(np.isnan(mat1.flatten()) | np.isnan(mat2.flatten()))
        if valid.sum() > 1:
            corr = np.corrcoef(mat1.flatten()[valid], mat2.flatten()[valid])[0, 1]
        else:
            corr = np.nan

        # RMSE
        rmse = np.sqrt(np.nanmean(diff ** 2))

        # Check if identical within tolerance
        is_identical = max_diff <= args.tolerance

        status = "✓" if is_identical else "❌"
        print(f"  {status} Shape:       {mat1.shape}")
        print(f"    Max diff:    {max_diff:.2e}")
        print(f"    Mean diff:   {mean_diff:.2e}")
        print(f"    RMSE:        {rmse:.2e}")
        print(f"    Correlation: {corr:.10f}")

        if not is_identical:
            all_identical = False
            # Show where differences occur
            max_idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)
            print(f"    Max diff at: {max_idx}")
            print(f"    File 1 val:  {mat1[max_idx]:.10f}")
            print(f"    File 2 val:  {mat2[max_idx]:.10f}")

    # Compare sample/feature names
    print("\n" + "-" * 40)
    print("Index comparison:")

    # Get canonical names (cells and proteins) based on format
    if r_format1:
        cells1, proteins1 = data1.get('var_names', []), data1.get('obs_names', [])
    else:
        cells1, proteins1 = data1.get('obs_names', []), data1.get('var_names', [])

    if r_format2:
        cells2, proteins2 = data2.get('var_names', []), data2.get('obs_names', [])
    else:
        cells2, proteins2 = data2.get('obs_names', []), data2.get('var_names', [])

    # Compare cells
    if cells1 == cells2:
        print(f"  ✓ cells match ({len(cells1)} samples)")
    else:
        if set(cells1) == set(cells2):
            print(f"  ⚠ cells: same set, different order ({len(cells1)} samples)")
        else:
            common = set(cells1) & set(cells2)
            print(f"  ❌ cells differ: {len(cells1)} vs {len(cells2)}, {len(common)} common")
            if len(cells1) == len(cells2) == 1:
                print(f"       File 1: {cells1}")
                print(f"       File 2: {cells2}")
                if not getattr(args, 'by_position', False):
                    print("       Tip: Use --by-position to compare by position (ignoring cell names)")

    # Compare proteins
    if proteins1 == proteins2:
        print(f"  ✓ proteins match ({len(proteins1)} features)")
    else:
        if set(proteins1) == set(proteins2):
            print(f"  ⚠ proteins: same set, different order ({len(proteins1)} features)")
        else:
            common = set(proteins1) & set(proteins2)
            print(f"  ❌ proteins differ: {len(proteins1)} vs {len(proteins2)}, {len(common)} common")

    # Summary
    print("\n" + "=" * 60)
    if all_identical:
        print("✓ Files are IDENTICAL (within tolerance)")
        return 0
    else:
        print("❌ Files are DIFFERENT")
        return 1


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert R-format H5AD to Python-compatible format."""
    import numpy as np
    import h5py

    print("=" * 60)
    print("SecActPy - Convert R H5AD to Python Format")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("=" * 60)

    def read_matrix(f, path):
        """Read matrix from H5AD, handling both dense and sparse formats."""
        if path not in f:
            return None

        item = f[path]
        if isinstance(item, h5py.Dataset):
            # Dense matrix
            return item[:]
        elif isinstance(item, h5py.Group):
            # Sparse matrix (CSR/CSC format)
            if 'data' in item and 'indices' in item and 'indptr' in item:
                from scipy import sparse
                data = item['data'][:]
                indices = item['indices'][:]
                indptr = item['indptr'][:]
                shape = tuple(item.attrs.get('shape', []))

                # Determine format from encoding-type or shape
                encoding = item.attrs.get('encoding-type', b'')
                if isinstance(encoding, bytes):
                    encoding = encoding.decode('utf-8')

                if 'csr' in encoding.lower() or (shape and shape[0] == len(indptr) - 1):
                    sp_mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
                else:
                    sp_mat = sparse.csc_matrix((data, indices, indptr), shape=shape)

                return sp_mat.toarray()
        return None

    def get_index(f, group_name):
        """Get index from obs or var group, handling different formats."""
        if group_name not in f:
            return []
        grp = f[group_name]

        # Try _index first
        if '_index' in grp:
            idx = grp['_index'][:]
        # Try index attribute pointing to a dataset
        elif 'index' in grp.attrs:
            idx_name = grp.attrs['index']
            if isinstance(idx_name, bytes):
                idx_name = idx_name.decode('utf-8')
            if idx_name in grp:
                idx = grp[idx_name][:]
            else:
                idx = []
        else:
            # Look for any dataset that might be the index
            idx = []
            for key in grp.keys():
                if key.startswith('_') or key == 'index':
                    idx = grp[key][:]
                    break

        if len(idx) > 0:
            if isinstance(idx[0], bytes):
                idx = [x.decode('utf-8') for x in idx]
            return list(idx)
        return []

    # Read R-format H5AD using h5py directly
    with h5py.File(args.input, 'r') as f:
        # Get X matrix
        X = read_matrix(f, 'X')
        if X is None:
            print("Error: No X matrix found in file")
            return 1

        print(f"\nOriginal X shape: {X.shape}")

        # Get indices
        obs_names = get_index(f, 'obs')
        var_names = get_index(f, 'var')

        print(f"obs_names: {len(obs_names)}")
        print(f"var_names: {len(var_names)}")

        # Get obsm matrices
        obsm = {}
        if 'obsm' in f:
            for key in f['obsm'].keys():
                mat = read_matrix(f, 'obsm/' + key)
                if mat is not None:
                    obsm[key] = mat
                    print(f"obsm/{key}: {mat.shape}")

        # Detect if transposed (R format: proteins as obs, cells as var)
        # Python format: cells as obs, proteins as var
        is_transposed = False

        # Heuristic 1: Check shape vs indices
        if len(obs_names) > 0 and len(var_names) > 0:
            if X.shape[0] == len(var_names) and X.shape[1] == len(obs_names):
                is_transposed = True

        # Heuristic 2: If obs has few entries (proteins) and var has many (cells)
        if len(obs_names) < 2000 and len(var_names) > 10000:
            is_transposed = True

        if is_transposed:
            print("\nDetected R format (proteins as obs, cells as var)")
            print("Transposing to Python format (cells as obs, proteins as var)...")

            # Swap
            X = X.T
            obs_names, var_names = var_names, obs_names

            # Transpose obsm matrices too
            for key in obsm:
                obsm[key] = obsm[key].T

            print(f"New X shape: {X.shape}")
        else:
            print("\nFile appears to already be in Python format")

    # Write in Python-compatible format
    print(f"\nWriting to: {args.output}")

    with h5py.File(args.output, 'w') as f:
        # Create groups
        f.create_group('obs')
        f.create_group('var')
        f.create_group('obsm')
        f.create_group('uns')

        # Write X
        f.create_dataset('X', data=X, compression='gzip')

        # Write indices
        f.create_dataset('obs/_index', data=np.array(obs_names, dtype='S'))
        f.create_dataset('var/_index', data=np.array(var_names, dtype='S'))

        # Write obsm
        for key, arr in obsm.items():
            f.create_dataset('obsm/' + key, data=arr, compression='gzip')

        # Add anndata compatibility attributes
        f.attrs['encoding-type'] = 'anndata'
        f.attrs['encoding-version'] = '0.8.0'

        f['obs'].attrs['encoding-type'] = 'dataframe'
        f['obs'].attrs['encoding-version'] = '0.2.0'
        f['obs'].attrs['_index'] = '_index'
        f['obs'].attrs['column-order'] = []

        f['var'].attrs['encoding-type'] = 'dataframe'
        f['var'].attrs['encoding-version'] = '0.2.0'
        f['var'].attrs['_index'] = '_index'
        f['var'].attrs['column-order'] = []

    print("\nDone! File can now be loaded with:")
    print("  import anndata")
    print(f"  adata = anndata.read_h5ad('{args.output}')")
    print(f"  # Shape: ({len(obs_names)}, {len(var_names)}) = (cells × proteins)")

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect H5AD file structure."""
    import h5py

    print("=" * 60)
    print("SecActPy - H5AD Inspector")
    print("=" * 60)
    print(f"File: {args.input}")
    print("=" * 60)

    with h5py.File(args.input, 'r') as f:
        print("\nFile structure:")

        def print_item(name, obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{prefix}📁 {name}/")
                # Print attributes
                if obj.attrs:
                    for attr_name, attr_val in obj.attrs.items():
                        val_str = str(attr_val)[:50]
                        print(f"{prefix}   @{attr_name}: {val_str}")
            else:
                shape_str = str(obj.shape) if hasattr(obj, 'shape') else '?'
                dtype_str = str(obj.dtype) if hasattr(obj, 'dtype') else '?'
                print(f"{prefix}📄 {name}: {shape_str} ({dtype_str})")

        # Print top-level items
        for key in sorted(f.keys()):
            item = f[key]
            print_item(key, item, indent=0)

            if isinstance(item, h5py.Group):
                for subkey in sorted(item.keys()):
                    subitem = item[subkey]
                    print_item(subkey, subitem, indent=1)

                    if isinstance(subitem, h5py.Group) and args.verbose:
                        for subsubkey in sorted(subitem.keys()):
                            print_item(subsubkey, subitem[subsubkey], indent=2)

        # Summary
        print("\n" + "-" * 40)
        print("Summary:")

        # X shape
        if 'X' in f:
            x_item = f['X']
            if isinstance(x_item, h5py.Dataset):
                print(f"  X: {x_item.shape} (dense)")
            elif isinstance(x_item, h5py.Group):
                shape = x_item.attrs.get('shape', 'unknown')
                print(f"  X: {tuple(shape)} (sparse)")

        # obs/var counts
        def get_index_count(group_name):
            if group_name not in f:
                return 0
            grp = f[group_name]
            if '_index' in grp:
                return len(grp['_index'])
            return 0

        n_obs = get_index_count('obs')
        n_var = get_index_count('var')
        print(f"  obs (rows): {n_obs}")
        print(f"  var (cols): {n_var}")

        # obsm keys
        if 'obsm' in f:
            obsm_keys = list(f['obsm'].keys())
            print(f"  obsm keys: {obsm_keys}")

        # reductions keys
        if 'reductions' in f:
            red_keys = list(f['reductions'].keys())
            print(f"  reductions keys: {red_keys}")

        # layers keys
        if 'layers' in f:
            layer_keys = list(f['layers'].keys())
            print(f"  layers keys: {layer_keys}")

    return 0


def main(argv=None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="secactpy",
        description="SecActPy: Secreted protein activity inference from gene expression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bulk RNA-seq (differential expression)
  secactpy bulk -i diff_expr.tsv -o results.h5ad --differential -v

  # Bulk RNA-seq (raw counts, compute differential vs mean)
  secactpy bulk -i counts.tsv -o results.h5ad -v

  # scRNA-seq with cell type aggregation
  secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype -v

  # scRNA-seq at single cell level
  secactpy scrnaseq -i data.h5ad -o results.h5ad --single-cell -v

  # Visium spatial transcriptomics (10x format)
  secactpy visium -i /path/to/visium/ -o results.h5ad -v

  # Visium with cell type deconvolution
  secactpy visium -i data.h5ad -o results.h5ad --cell-type-col cell_type -v

  # CosMx (single-cell spatial)
  secactpy cosmx -i cosmx.h5ad -o results.h5ad --batch-size 50000 -v

  # Use GPU acceleration
  secactpy bulk -i data.tsv -o results.h5ad --backend cupy -v

  # Use CytoSig signature
  secactpy bulk -i data.tsv -o results.h5ad --signature cytosig -v

  # Compare R and Python outputs
  secactpy compare r_output.h5ad python_output.h5ad -t 1e-10

  # Compare with different index ordering
  secactpy compare file1.h5ad file2.h5ad --sort -v

  # Convert R-format H5AD to Python-compatible format
  secactpy convert -i r_output.h5ad -o python_compatible.h5ad

  # Inspect H5AD file structure (for debugging)
  secactpy inspect -i file.h5ad -v

For more information, visit: https://github.com/data2intelligence/SecActpy
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0"
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>"
    )

    # Bulk RNA-seq
    bulk_parser = subparsers.add_parser(
        "bulk",
        help="Bulk RNA-seq inference",
        description="Infer secreted protein activity from bulk RNA-seq data"
    )
    setup_common_args(bulk_parser)
    bulk_parser.add_argument(
        "-d", "--differential",
        action="store_true",
        help="Input is already differential expression (log2FC)"
    )
    bulk_parser.add_argument(
        "--sig-filter",
        action="store_true",
        help="Filter signatures by available genes"
    )
    bulk_parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose H5AD input (if genes are in obs instead of var)"
    )
    bulk_parser.set_defaults(func=cmd_bulk)

    # scRNA-seq
    scrna_parser = subparsers.add_parser(
        "scrnaseq",
        help="scRNA-seq inference",
        description="Infer secreted protein activity from scRNA-seq data"
    )
    setup_common_args(scrna_parser)
    scrna_parser.add_argument(
        "-c", "--cell-type-col",
        default=None,
        help="Column name for cell type annotations"
    )
    scrna_parser.add_argument(
        "--single-cell",
        action="store_true",
        help="Compute at single cell level (default: aggregate by cell type)"
    )
    scrna_parser.add_argument(
        "--sig-filter",
        action="store_true",
        help="Filter signatures by available genes"
    )
    scrna_parser.set_defaults(func=cmd_scrnaseq)

    # Visium
    visium_parser = subparsers.add_parser(
        "visium",
        help="Visium spatial transcriptomics inference",
        description="Infer secreted protein activity from Visium data"
    )
    setup_common_args(visium_parser)
    visium_parser.add_argument(
        "-c", "--cell-type-col",
        default=None,
        help="Column name for cell type deconvolution results"
    )
    visium_parser.add_argument(
        "--spot-level",
        action="store_true",
        default=True,
        help="Compute at spot level (default: True)"
    )
    visium_parser.add_argument(
        "--min-genes",
        type=int,
        default=200,
        help="Minimum genes per spot (default: 200)"
    )
    visium_parser.add_argument(
        "--scale-factor",
        type=float,
        default=1e5,
        help="Normalization scale factor (default: 1e5)"
    )
    visium_parser.add_argument(
        "--sig-filter",
        action="store_true",
        help="Filter signatures by available genes"
    )
    visium_parser.set_defaults(func=cmd_visium)

    # CosMx
    cosmx_parser = subparsers.add_parser(
        "cosmx",
        help="CosMx spatial transcriptomics inference",
        description="Infer secreted protein activity from CosMx data"
    )
    setup_common_args(cosmx_parser)
    cosmx_parser.add_argument(
        "-c", "--cell-type-col",
        default=None,
        help="Column name for cell type annotations"
    )
    cosmx_parser.add_argument(
        "--min-genes",
        type=int,
        default=50,
        help="Minimum genes per cell (default: 50)"
    )
    cosmx_parser.add_argument(
        "--scale-factor",
        type=float,
        default=1000,
        help="Normalization scale factor (default: 1000 for CosMx)"
    )
    cosmx_parser.add_argument(
        "--sig-filter",
        action="store_true",
        default=True,
        help="Filter signatures by available genes (default: True for CosMx)"
    )
    cosmx_parser.add_argument(
        "--no-sig-filter",
        action="store_false",
        dest="sig_filter",
        help="Disable signature filtering"
    )
    cosmx_parser.set_defaults(func=cmd_cosmx)

    # Compare H5AD files
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two H5AD files for validation",
        description="Compare two H5AD files to validate R vs Python output identity"
    )
    compare_parser.add_argument(
        "file1",
        help="First H5AD file (e.g., R output)"
    )
    compare_parser.add_argument(
        "file2",
        help="Second H5AD file (e.g., Python output)"
    )
    compare_parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=1e-10,
        help="Tolerance for numerical comparison (default: 1e-10)"
    )
    compare_parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort rows/columns before comparison (for different ordering)"
    )
    compare_parser.add_argument(
        "--by-position",
        action="store_true",
        help="Compare by matrix position, ignoring cell/protein names (useful for single-sample)"
    )
    compare_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Convert R H5AD to Python format
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert R-format H5AD to Python-compatible format",
        description="Convert H5AD files written by R to Python/anndata-compatible format"
    )
    convert_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input H5AD file (R format)"
    )
    convert_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output H5AD file (Python format)"
    )
    convert_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    convert_parser.set_defaults(func=cmd_convert)

    # Inspect H5AD file structure
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect H5AD file structure",
        description="Show the internal structure of an H5AD file for debugging"
    )
    inspect_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input H5AD file to inspect"
    )
    inspect_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show nested structure details"
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    # Parse arguments
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
