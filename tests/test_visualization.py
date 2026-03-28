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
    spatial_density,
)


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
