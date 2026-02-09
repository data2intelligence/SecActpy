# Command Line Interface

SecActPy provides a command line interface for common workflows.

## Commands

```bash
# Bulk RNA-seq (differential expression)
secactpy bulk -i diff_expr.tsv -o results.h5ad --differential -v

# Bulk RNA-seq (raw counts)
secactpy bulk -i counts.tsv -o results.h5ad -v

# scRNA-seq with cell type aggregation
secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype -v

# scRNA-seq at single cell level
secactpy scrnaseq -i data.h5ad -o results.h5ad --single-cell -v

# Visium spatial transcriptomics
secactpy visium -i /path/to/visium/ -o results.h5ad -v

# CosMx (single-cell spatial)
secactpy cosmx -i cosmx.h5ad -o results.h5ad --batch-size 50000 -v

# Use GPU acceleration
secactpy bulk -i data.tsv -o results.h5ad --backend cupy -v

# Use CytoSig signature
secactpy bulk -i data.tsv -o results.h5ad --signature cytosig -v
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file or directory |
| `-o, --output` | Output H5AD file |
| `-s, --signature` | Signature matrix (secact, cytosig) |
| `--lambda` | Ridge regularization (default: 5e5) |
| `-n, --n-rand` | Number of permutations (default: 1000) |
| `--backend` | Computation backend (auto, numpy, cupy) |
| `--batch-size` | Batch size for large datasets |
| `-v, --verbose` | Verbose output |
