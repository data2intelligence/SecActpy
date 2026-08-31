"""Tests for secactpy.visualization module."""

import numpy as np
import pandas as pd
import pytest

from secactpy.visualization import (
    activity_correlation,
    activity_distribution,
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
