"""Tests for secactpy.visualization module."""

import numpy as np
import pandas as pd
import pytest

from secactpy.visualization import (
    activity_correlation,
    activity_distribution,
    ccc_heatmap,
    secreted_protein_dotplot,
    secreted_protein_split_heatmap,
    celltype_activity_boxplot,
    celltype_distribution,
    celltype_expression_boxplot,
    gene_expression_stats,
    secreted_protein_heatmap,
    spatial_density,
)
from secactpy.visualization import _SECACT_HEATMAP_COLORS


@pytest.fixture
def sample_activity():
    rng = np.random.default_rng(42)
    proteins = [f"P{i}" for i in range(5)]
    spots = [f"spot_{i}" for i in range(50)]
    data = rng.standard_normal((5, 50))
    return pd.DataFrame(data, index=proteins, columns=spots)


@pytest.fixture
def sample_cell_types():
    rng = np.random.default_rng(42)
    spots = [f"spot_{i}" for i in range(50)]
    types = rng.choice(["T-cell", "macrophage", "tumor", "fibroblast"], size=50)
    return pd.Series(types, index=spots)


@pytest.fixture
def sample_coordinates():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.uniform(0, 1000, 50),
        "y": rng.uniform(0, 1000, 50),
    })


def test_activity_distribution(sample_activity):
    fig = activity_distribution({"0": sample_activity, "10": sample_activity}, "P0")
    assert fig is not None
    assert len(fig.data) == 2


def test_activity_distribution_missing_protein(sample_activity):
    fig = activity_distribution({"0": sample_activity}, "MISSING")
    assert len(fig.data) == 0


def test_celltype_activity_boxplot(sample_activity, sample_cell_types):
    fig = celltype_activity_boxplot(sample_activity, sample_cell_types, "P0")
    assert fig is not None
    assert len(fig.data) == 4


def test_celltype_activity_boxplot_missing(sample_activity, sample_cell_types):
    fig = celltype_activity_boxplot(sample_activity, sample_cell_types, "MISSING")
    assert "not found" in fig.layout.annotations[0].text


def test_activity_correlation(sample_activity):
    target = sample_activity.copy()
    radii = {"10": sample_activity.copy(), "20": sample_activity.copy()}
    fig = activity_correlation(target, radii, "P0")
    assert fig is not None
    assert len(fig.data) >= 2


def test_gene_expression_stats(sample_activity):
    fig = gene_expression_stats(sample_activity)
    assert fig is not None
    assert len(fig.data) == 1


def test_celltype_expression_boxplot(sample_activity, sample_cell_types):
    fig = celltype_expression_boxplot(sample_activity, sample_cell_types, "P0")
    assert fig is not None


def test_celltype_distribution(sample_cell_types):
    fig = celltype_distribution(sample_cell_types)
    assert fig is not None
    assert len(fig.data[0].x) == 4


def test_spatial_density(sample_coordinates):
    fig = spatial_density(sample_coordinates)
    assert fig is not None
    assert len(fig.data) == 2


@pytest.fixture
def sample_celltype_activity():
    """Proteins x cell-types activity z-score matrix (like res['zscore'])."""
    rng = np.random.default_rng(1)
    proteins = [f"P{i}" for i in range(20)]
    cts = ["CD4_naive", "CD4_central_memory", "CD4_Th1_like"]
    return pd.DataFrame(rng.standard_normal((20, 3)), index=proteins, columns=cts)


def test_secreted_protein_heatmap_basic(sample_celltype_activity):
    mat = sample_celltype_activity
    fig = secreted_protein_heatmap(mat, title="Activity")
    h = fig.data[0]
    assert h.type == "heatmap"
    # columns preserved; rows reversed so the first matrix row is at the top
    assert list(h.x) == list(mat.columns)
    assert list(h.y)[-1] == mat.index[0] and list(h.y)[0] == mat.index[-1]
    assert h.z.shape == mat.shape
    np.testing.assert_allclose(h.z[-1], mat.iloc[0].values)   # top row == first protein
    # SecAct gradient, white gaps, rotated x labels, centered title
    assert [c[1] for c in h.colorscale] == _SECACT_HEATMAP_COLORS
    assert h.xgap == 1 and h.ygap == 1
    assert fig.layout.xaxis.tickangle == 90
    assert fig.layout.title.x == 0.5


def test_secreted_protein_heatmap_top_n(sample_celltype_activity):
    mat = sample_celltype_activity
    fig = secreted_protein_heatmap(mat, top_n=5)
    kept = set(fig.data[0].y)
    expected = set()
    for ct in mat.columns:
        expected |= set(mat[ct].sort_values(ascending=False).head(5).index)
    assert kept == expected            # union of top-5 per cell type
    assert len(kept) <= 5 * mat.shape[1]


def test_secreted_protein_heatmap_empty():
    fig = secreted_protein_heatmap(pd.DataFrame())
    assert fig is not None
    assert not fig.data or fig.data[0].type != "heatmap"


def test_ccc_heatmap_counts():
    ccc = pd.DataFrame({
        "sender":   ["A", "A", "B", "A", "C"],
        "receiver": ["B", "B", "C", "C", "A"],
        "secretedProtein": ["IL15", "TGFB1", "IL15", "CXCL9", "IL2"],
    })
    fig = ccc_heatmap(ccc)
    h = fig.data[0]
    assert h.type == "heatmap"
    # axes are the union of senders and receivers, aligned
    assert list(h.x) == ["A", "B", "C"]
    assert list(h.y) == ["C", "B", "A"]          # rows reversed (first sender on top)
    # A->B has 2 edges; z rows are reversed, so A is the last z-row
    zdf = pd.DataFrame(h.z, index=list(h.y), columns=list(h.x))
    assert zdf.loc["A", "B"] == 2
    assert zdf.loc["B", "C"] == 1 and zdf.loc["C", "A"] == 1
    assert fig.layout.yaxis.title.text == "Sender"
    assert fig.layout.xaxis.title.text == "Receiver"


def test_secreted_protein_dotplot_sc_false_se():
    """sc=False: color = activity, size = SE (cell-type level)."""
    rng = np.random.default_rng(0)
    prot = [f"P{i}" for i in range(12)]
    cts = ["A", "B", "C"]
    z = pd.DataFrame(rng.standard_normal((12, 3)), index=prot, columns=cts)
    se = pd.DataFrame(rng.uniform(0.1, 2, (12, 3)), index=prot, columns=cts)

    fig = secreted_protein_dotplot(z, sc=False, se=se, title="dot")
    tr = fig.data[0]
    assert tr.type == "scatter" and tr.mode == "markers"
    assert len(tr.x) == 12 * 3
    assert tr.marker.cmid == 0 and tr.marker.colorbar.title.text == "activity"
    assert len(set(np.round(tr.marker.size, 3))) > 1   # size varies with SE
    assert max(tr.y) == 11                             # first protein on top

    # no SE -> uniform dot size
    assert len(set(np.round(secreted_protein_dotplot(z).data[0].marker.size, 3))) == 1
    # empty
    empty = secreted_protein_dotplot(pd.DataFrame())
    assert not empty.data or empty.data[0].type != "scatter"


def test_secreted_protein_dotplot_sc_true_percell_sd():
    """sc=True: per-cell activity + cell_types -> color = mean, size = per-cell SD."""
    rng = np.random.default_rng(2)
    prot = [f"P{i}" for i in range(12)]
    cells = [f"c{i}" for i in range(150)]
    pca = pd.DataFrame(rng.standard_normal((12, 150)), index=prot, columns=cells)
    ct = pd.Series(["A"] * 60 + ["B"] * 50 + ["C"] * 40, index=cells)

    fig = secreted_protein_dotplot(pca, sc=True, cell_types=ct, spread="sd", min_cells=30)
    tr = fig.data[0]
    assert set(tr.x) == {"A", "B", "C"} and len(tr.x) == 12 * 3
    assert len(set(np.round(tr.marker.size, 3))) > 1   # size varies with per-cell SD
    # cell_types required when sc=True
    with pytest.raises(ValueError):
        secreted_protein_dotplot(pca, sc=True)


def test_secreted_protein_split_heatmap_sc_true():
    """sc=True: bottom-right = mean, top-left = per-cell spread."""
    rng = np.random.default_rng(1)
    prot = [f"P{i}" for i in range(15)]
    cells = [f"c{i}" for i in range(220)]
    pca = pd.DataFrame(rng.standard_normal((15, 220)), index=prot, columns=cells)
    ct = pd.Series(["A"] * 90 + ["B"] * 70 + ["C"] * 50 + ["D"] * 10, index=cells)

    fig = secreted_protein_split_heatmap(pca, sc=True, cell_types=ct, spread="sd",
                                         min_cells=30, top_n=5, title="split")
    assert set(fig.layout.xaxis.ticktext) == {"A", "B", "C"}   # D gated (< min_cells)
    assert len(fig.layout.shapes) > 0 and len(fig.layout.shapes) % 2 == 0
    assert len([t for t in fig.data if getattr(t.marker, "showscale", False)]) == 2
    for s in ("sd", "cv", "iqr"):
        assert secreted_protein_split_heatmap(pca, sc=True, cell_types=ct,
                                              spread=s, min_cells=30).layout.shapes
    # nothing passes the cell-count floor -> placeholder
    assert secreted_protein_split_heatmap(pca, sc=True, cell_types=ct,
                                          min_cells=300).layout.shapes == ()


def test_secreted_protein_split_heatmap_sc_false_se():
    """sc=False: bottom-right = activity, top-left = SE."""
    rng = np.random.default_rng(3)
    prot = [f"P{i}" for i in range(15)]
    z = pd.DataFrame(rng.standard_normal((15, 3)), index=prot, columns=["A", "B", "C"])
    se = pd.DataFrame(rng.uniform(0.1, 2, (15, 3)), index=prot, columns=["A", "B", "C"])

    fig = secreted_protein_split_heatmap(z, sc=False, se=se, top_n=5)
    assert len(fig.layout.shapes) > 0 and len(fig.layout.shapes) % 2 == 0
    assert len([t for t in fig.data if getattr(t.marker, "showscale", False)]) == 2
    # sc=False needs se for the top-left triangle
    with pytest.raises(ValueError):
        secreted_protein_split_heatmap(z, sc=False)


def test_ccc_heatmap_accepts_result_dict_and_empty():
    # full result dict
    fig = ccc_heatmap({"secreted_protein_ccc": pd.DataFrame(
        {"sender": ["A"], "receiver": ["B"], "secretedProtein": ["IL15"]})})
    assert fig.data[0].type == "heatmap"
    # empty / missing columns -> placeholder figure (no heatmap trace), not a crash
    empty = ccc_heatmap(pd.DataFrame())
    assert not empty.data or empty.data[0].type != "heatmap"
    empty2 = ccc_heatmap({"secreted_protein_ccc": pd.DataFrame()})
    assert not empty2.data or empty2.data[0].type != "heatmap"
