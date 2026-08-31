"""Tests for secactpy.ccc (cell-cell communication inference)."""

import numpy as np
import pandas as pd
import pytest

from secactpy.ccc import _benjamini_hochberg, _normalize_log, secact_ccc_scrnaseq


def test_benjamini_hochberg_matches_reference():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    bh = _benjamini_hochberg(p)
    assert bh.max() <= 1.0
    assert np.all(np.diff(bh) >= -1e-12)  # monotone non-decreasing in sorted order
    fdc = pytest.importorskip("scipy.stats").false_discovery_control
    np.testing.assert_allclose(bh, fdc(p, method="bh"), atol=1e-12)


def test_benjamini_hochberg_handles_nan():
    p = np.array([0.01, np.nan, 0.5])
    bh = _benjamini_hochberg(p)
    assert np.isnan(bh[1])
    assert not np.isnan(bh[0]) and not np.isnan(bh[2])


def test_normalize_log_per_cell():
    block = np.array([[0.0, 2.0, 4.0], [1.0, 0.0, 0.0]])
    out = _normalize_log(block, 1e5)
    assert out.shape == block.shape
    assert np.all(out >= 0)  # log2(x+1) with x>=0
    # a zero-count cell stays all-zero (guarded division)
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0


@pytest.fixture
def synthetic_adata():
    anndata = pytest.importorskip("anndata")
    from secactpy.signature import load_signature

    sig = load_signature("secact")
    genes_feat = [str(g).upper() for g in sig.index[:500]]
    sp_names = {str(c).upper() for c in sig.columns}
    extra_sp = [s for s in sp_names if s not in genes_feat][:6]
    genes = list(dict.fromkeys(genes_feat + extra_sp))
    gi = {g: i for i, g in enumerate(genes)}
    sp_measurable = [g for g in genes if g in sp_names]

    rng = np.random.default_rng(0)
    cts = np.repeat(["A", "B", "C"], 80)
    conds = np.tile(np.repeat(["Metastatic", "Primary"], 40), 3)
    n = cts.size
    X = rng.poisson(1.0, size=(n, len(genes))).astype(float)
    # a sender: boost several secreted-protein genes in A / Metastatic
    for s in sp_measurable[:3]:
        m = (cts == "A") & (conds == "Metastatic")
        X[m, gi[s]] += rng.poisson(8.0, size=m.sum())

    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"MyCellType": cts, "Groups": conds},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=genes),
    )


def test_secact_ccc_scrnaseq_structure(synthetic_adata):
    res = secact_ccc_scrnaseq(
        synthetic_adata, cell_type_col="MyCellType", condition_col="Groups",
        condition_case="Metastatic", condition_control="Primary",
        n_rand=100, min_cells=30, act_diff_cutoff=0.5, padj_cutoff=0.5,
        exp_logfc_cutoff=0.0, exp_mean_all_cutoff=0.0, exp_fraction_case_cutoff=0.0,
        backend="numpy", seed=0,
    )
    assert set(res) == {
        "secreted_protein_expression",
        "secreted_protein_activity",
        "secreted_protein_ccc",
    }
    exp = res["secreted_protein_expression"]
    assert len(exp) >= 2
    for df in exp.values():
        assert {"exp_logFC", "exp_mean_all", "exp_fraction_case",
                "exp_pv", "exp_pv.adj"}.issubset(df.columns)

    act = res["secreted_protein_activity"]
    assert "zscore" in act and "pvalue" in act
    assert list(act["zscore"].columns) == list(exp.keys())

    ccc = res["secreted_protein_ccc"]
    assert isinstance(ccc, pd.DataFrame)
    if not ccc.empty:
        assert {"sender", "secretedProtein", "receiver",
                "overall_strength", "overall_pv.adj"}.issubset(ccc.columns)
        # never self-communication
        assert (ccc["sender"] != ccc["receiver"]).all()


def test_secact_ccc_requires_condition_labels(synthetic_adata):
    with pytest.raises(ValueError):
        secact_ccc_scrnaseq(synthetic_adata, cell_type_col="MyCellType",
                            condition_col="Groups")  # missing case/control
