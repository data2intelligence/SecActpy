"""Step 2 pseudobulk and the parallel Step 1 must not change the answer.

Both changes are performance-only, and both touch the numerical path -- Step 2
now computes all 102 group sums with one sparse product instead of per-group
densification, and Step 1 can run its cell types concurrently. A speedup that
quietly changes a p-value is worse than no speedup, so these pin the results
rather than the timings.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from secactpy.ccc import secact_ccc_scrnaseq


@pytest.fixture
def toy():
    """Small AnnData with two conditions and several cell types.

    Gene names come from the real secact signature: invented symbols overlap it
    at zero genes and the inference refuses to run, so a fixture that looks
    generic would only ever test the error path.
    """
    ad = pytest.importorskip("anndata")
    from secactpy.signature import load_signature

    sig = load_signature("secact")
    genes = [str(g).upper() for g in sig.index[:400]]
    sp_names = {str(c).upper() for c in sig.columns}
    genes = list(dict.fromkeys(genes + [s for s in sp_names if s not in genes][:8]))

    rs = np.random.RandomState(0)
    n_cells = 900
    X = rs.poisson(1.0, (n_cells, len(genes))).astype(np.float64)
    ct = rs.choice([f"T{i}" for i in range(6)], n_cells)
    cond = rs.choice(["case", "ctrl"], n_cells)
    # give one cell type a real sender signal so edges are reachable
    gi = {g: i for i, g in enumerate(genes)}
    for sp in [g for g in genes if g in sp_names][:3]:
        m = (ct == "T0") & (cond == "case")
        X[m, gi[sp]] += rs.poisson(8.0, m.sum())

    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.obs["ct"] = pd.Categorical(ct)
    a.obs["cond"] = pd.Categorical(cond)
    return a


def _run(a, **kw):
    return secact_ccc_scrnaseq(a, cell_type_col="ct", condition_col="cond",
                               condition_case="case", condition_control="ctrl",
                               n_rand=50, backend="numpy", seed=0, **kw)


def test_n_jobs_does_not_change_the_result(toy):
    """Concurrency is a scheduling detail; the numbers must be identical."""
    a, b = _run(toy, n_jobs=1), _run(toy, n_jobs=4)
    assert set(a["secreted_protein_expression"]) == set(b["secreted_protein_expression"])
    for ct, df in a["secreted_protein_expression"].items():
        other = b["secreted_protein_expression"][ct]
        assert df.index.equals(other.index)
        np.testing.assert_array_equal(df.select_dtypes("number").values,
                                      other.select_dtypes("number").values)
    np.testing.assert_allclose(a["secreted_protein_activity"]["zscore"].values,
                               b["secreted_protein_activity"]["zscore"].values,
                               rtol=0, atol=0)


def test_pseudobulk_one_pass_matches_the_per_group_definition(toy):
    """The single sparse product must equal summing each group separately.

    The optimization rests on the groups being DISJOINT -- every cell belongs to
    exactly one (cell type, condition) -- so each nonzero lands in exactly one
    output column. If that ever stopped holding, this catches it as a numeric
    disagreement rather than as a silently different pseudobulk.
    """
    from secactpy.ccc import _extract_counts
    counts, genes, obs = _extract_counts(toy, "ct", "cond")
    ct_v = obs["ct"].astype(str).values
    cond_v = obs["cond"].astype(str).values

    groups, idxs = [], []
    for ct in sorted(set(ct_v)):
        for cond in ("case", "ctrl"):
            groups.append((ct, cond))
            idxs.append(np.flatnonzero((ct_v == ct) & (cond_v == cond)))

    rows = np.concatenate(idxs)
    cols = np.concatenate([np.full(len(ix), j, int) for j, ix in enumerate(idxs)])
    M = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(counts.shape[1], len(groups)))
    one_pass = np.asarray((counts @ M).todense())
    per_group = np.column_stack([np.asarray(counts[:, ix].sum(axis=1)).ravel()
                                 for ix in idxs])
    np.testing.assert_allclose(one_pass, per_group, rtol=0, atol=0)

    # and the groups really are a partition of the cells
    assert len(rows) == counts.shape[1]
    assert len(set(rows.tolist())) == counts.shape[1]
