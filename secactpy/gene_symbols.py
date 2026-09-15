"""Gene-symbol normalisation, matching SecAct R's `transferSymbol` exactly.

SecAct R maps each gene name through an NCBI alias->symbol table before matching it
against the signature matrix (`SecAct/R/ifun.R`, applied at every entry point in
`activity.R` and `downstream.R`). secactpy previously uppercased the names and stripped
version suffixes instead, under a comment claiming it matched R. It does not: R does
neither of those things, and does a lookup secactpy did not do at all.

The consequences were measured on a 500-cell TdLN subset against SecAct 1.0.0:

  * no alias lookup  -- R resolves 123 of 19,661 names and gains 45 genes in the
    signature intersection, so the two implementations fit different design matrices
  * uppercasing      -- changes 173 names, and 0.7% of SecAct's own signature genes are
    NOT uppercase (`C1orf159`, `C22orf15`), so uppercasing the data silently loses them
  * version stripping -- changes 11 names, with no counterpart in R

Together these left beta differing by up to 1.0e-4 where the project's cross-language
target is 1e-10.

R's semantics, reproduced here rather than approximated:

  * the table is read with the first two comma-separated fields per line; 8 rows carry a
    third field which R's `read.csv` drops (`NERF-1a,b,ELF2` -> ("NERF-1a", "b"))
  * a missing Alias becomes the literal string "NA"
  * `match()` returns the FIRST hit, so for the 7 duplicated alias keys the first row wins
  * a name absent from the table is left untouched -- and case is significant throughout
"""
from __future__ import annotations
import gzip, os
from functools import lru_cache
from typing import Iterable, Sequence
import numpy as np

_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "NCBI_20251008_gene_result_alias2symbol.csv.gz")


@lru_cache(maxsize=1)
def _alias_map(path: str = _TABLE) -> dict:
    """alias -> symbol, first occurrence winning, as R's match() does."""
    out: dict[str, str] = {}
    with gzip.open(path, "rt") as fh:
        header = fh.readline()
        if not header.lower().startswith("alias"):
            raise ValueError(f"unexpected header in {path}: {header!r}")
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split(",")
            alias = parts[0] if parts[0] != "" else "NA"      # R fills the NA alias
            symbol = parts[1] if len(parts) > 1 else ""
            out.setdefault(alias, symbol)                      # first hit wins
    return out


def transfer_symbol(names: Sequence[str] | Iterable[str], table: str = _TABLE) -> np.ndarray:
    """Map aliases to current symbols. Names absent from the table pass through.

    Case-sensitive and version-preserving, because R is both.
    """
    amap = _alias_map(table)
    return np.array([amap.get(str(g), str(g)) for g in names], dtype=object)


def rm_duplicates(names, mat=None, row_sums=None):
    """Indices to keep, mirroring SecAct R's `rm_duplicates` (ifun.R:35).

    R's rule, reproduced exactly:
      * every uniquely-named row is kept
      * for each duplicated name, keep the row with the largest rowSums; on a tie take
        the FIRST such row (`max_flag[1]`)
      * the survivors are returned in ascending ORIGINAL row order
        (`mat[sort(c(unique_idx, dupl_idx)), ]`), not alphabetical

    The row order matters beyond bookkeeping: SecAct later takes
    `intersect(rownames(Y), rownames(X))`, so Y's row order sets the order of the genes
    fed to the ridge solve, and that sets the summation order of the reduction.

    Pass either `mat` (genes x cells, dense or sparse) or a precomputed `row_sums`.
    """
    import numpy as np
    names = np.asarray([str(x) for x in names], dtype=object)
    if row_sums is None:
        if mat is None:
            raise ValueError("pass mat or row_sums")
        row_sums = (np.asarray(mat.sum(axis=1)).ravel() if hasattr(mat, "tocsr")
                    else np.asarray(mat).sum(axis=1))
    row_sums = np.asarray(row_sums).ravel()

    first, counts = {}, {}
    for i, g in enumerate(names):
        counts[g] = counts.get(g, 0) + 1
    keep = []
    best = {}
    for i, g in enumerate(names):
        if counts[g] == 1:
            keep.append(i)
        else:
            b = best.get(g)
            if b is None or row_sums[i] > row_sums[b]:   # strict > keeps the first max
                best[g] = i
    keep.extend(best.values())
    return np.array(sorted(keep), dtype=int)
