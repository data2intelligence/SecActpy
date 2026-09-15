"""secactpy must resolve exactly the genes SecAct R resolves.

R runs `transferSymbol()` -- an NCBI alias->symbol lookup shipped with the package -- then
`rm_duplicates()`. secactpy previously did something else entirely under a comment
claiming it matched: it uppercased names and stripped version suffixes, with no lookup.
The consequences on a 500-cell TdLN subset were

    after alias    R 19,661   py 19,661
    after dedup    R 19,658   py 19,655      <- 3 genes
    overlap w/sig  R  7,873   py  7,774      <- 99 genes

so the two fitted different design matrices, which is why beta differed by ~1e-4 while the
ridge kernels themselves agree to ~1e-16.

Uppercasing was independently wrong: 0.7% of SecAct's own signature genes are not
uppercase (C1orf159, C22orf15), so uppercasing the data silently drops them.

ORDER is asserted as well as membership. SecAct takes
`intersect(rownames(Y), rownames(X))`, so Y's row order decides the gene order handed to
the solve, and therefore the summation order of a reduction over thousands of terms.
Same set in a different order would still shift the last bits.
"""
from __future__ import annotations
import os
import numpy as np
import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "gene_resolution_r.npz")


@pytest.fixture(scope="module")
def fx():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture missing: {FIXTURE}")
    return np.load(FIXTURE, allow_pickle=True)


def test_transfer_symbol_matches_r(fx):
    from secactpy.gene_symbols import transfer_symbol
    got = np.asarray(transfer_symbol(fx["genes_in"]), dtype=object)
    ref = fx["r_after_alias"]
    assert len(got) == len(ref)
    bad = np.flatnonzero(got != ref)
    assert bad.size == 0, (
        f"{bad.size} names differ from R's transferSymbol, e.g. "
        f"{[(fx['genes_in'][i], got[i], ref[i]) for i in bad[:5]]}")


def test_rm_duplicates_matches_r_including_order(fx):
    from secactpy.gene_symbols import transfer_symbol, rm_duplicates
    names = np.asarray(transfer_symbol(fx["genes_in"]), dtype=object)
    keep = rm_duplicates(names, row_sums=fx["row_sums"])
    got, ref = names[keep], fx["r_after_dedup"]
    assert len(got) == len(ref), f"kept {len(got)} rows, R kept {len(ref)}"
    assert np.all(got == ref), "same count but different rows or different order than R"


def test_signature_overlap_matches_r(fx):
    """The genes actually handed to the ridge, in the order they are handed."""
    import pandas as pd
    from secactpy.gene_symbols import transfer_symbol, rm_duplicates
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "secactpy", "data", "SecAct.tsv.gz")
    if not os.path.exists(data):
        pytest.skip("SecAct signature matrix not present")
    sg = set(map(str, pd.read_csv(data, sep="\t", index_col=0, usecols=[0]).index))
    names = np.asarray(transfer_symbol(fx["genes_in"]), dtype=object)
    kept = names[rm_duplicates(names, row_sums=fx["row_sums"])]
    got = np.array([g for g in kept if g in sg], dtype=object)
    ref = fx["r_overlap"]
    assert len(got) == len(ref), f"overlap {len(got)} vs R {len(ref)}"
    assert np.all(got == ref), "overlap membership or order differs from R"


def test_uppercasing_would_lose_signature_genes():
    """Pins why the old behaviour was wrong, so it is not reintroduced as a tidy-up."""
    import pandas as pd
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "secactpy", "data", "SecAct.tsv.gz")
    if not os.path.exists(data):
        pytest.skip("SecAct signature matrix not present")
    g = pd.Index(map(str, pd.read_csv(data, sep="\t", index_col=0, usecols=[0]).index))
    not_upper = g[g != g.str.upper()]
    assert len(not_upper) > 0, (
        "no mixed-case signature genes remain; if SecAct normalised its matrix this "
        "test can go, but check transfer_symbol first")
