"""
SecAct-specific analysis plots.

Pure functions returning Plotly figures — usable in scripts, notebooks, and the Dash app.
Mirrors R's SecActViz-R statistics plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "activity_distribution",
    "celltype_activity_boxplot",
    "activity_correlation",
    "gene_expression_stats",
    "celltype_expression_boxplot",
    "celltype_distribution",
    "spatial_density",
]

_PRIMARY = "#3498db"
_ACCENT = "#e74c3c"


def activity_distribution(
    activity_dict: dict[str, pd.DataFrame],
    protein: str,
) -> go.Figure:
    """Violin + boxplot of protein activity across radii.

    Parameters
    ----------
    activity_dict : dict[str, DataFrame]
        Mapping from radius label (e.g. "0", "10", "20") to activity matrix
        (proteins x spots). Key "0" is the target.
    protein : str
        Protein name (must be a row in each matrix).
    """
    fig = go.Figure()
    for radius, mat in sorted(activity_dict.items(), key=lambda x: float(x[0])):
        if protein not in mat.index:
            continue
        values = mat.loc[protein].dropna().values.astype(float)
        label = "Target" if radius == "0" else f"{radius} \u03bcm"
        fig.add_trace(go.Violin(
            y=values, name=label, box_visible=True,
            meanline_visible=True, fillcolor=_PRIMARY, opacity=0.7,
            line_color=_PRIMARY,
        ))
    fig.update_layout(
        title=f"{protein} Activity Distribution",
        yaxis_title="Activity",
        showlegend=False,
        template="plotly_white",
    )
    return fig


def celltype_activity_boxplot(
    activity_matrix: pd.DataFrame,
    cell_types: pd.Series,
    protein: str,
) -> go.Figure:
    """Boxplot of activity by cell type, ordered by median.

    Parameters
    ----------
    activity_matrix : DataFrame
        Proteins x spots.
    cell_types : Series
        Cell type label per spot, indexed by spot ID.
    protein : str
        Protein to plot.
    """
    if protein not in activity_matrix.index:
        return _empty_figure(f"Protein {protein} not found")
    values = activity_matrix.loc[protein]
    common = values.index.intersection(cell_types.index)
    df = pd.DataFrame({"activity": values[common], "celltype": cell_types[common]})
    df = df.dropna()

    order = df.groupby("celltype")["activity"].median().sort_values(ascending=False).index

    fig = go.Figure()
    for ct in order:
        subset = df[df["celltype"] == ct]["activity"]
        fig.add_trace(go.Box(y=subset, name=ct, boxmean=True))
    fig.update_layout(
        title=f"{protein} Activity by Cell Type",
        yaxis_title="Activity",
        showlegend=False,
        template="plotly_white",
    )
    return fig


def activity_correlation(
    target_activity: pd.DataFrame,
    radius_activities: dict[str, pd.DataFrame],
    protein: str,
) -> go.Figure:
    """Scatter + regression: target vs each radius activity.

    Parameters
    ----------
    target_activity : DataFrame
        Target (radius 0) activity matrix (proteins x spots).
    radius_activities : dict[str, DataFrame]
        Non-zero radius activity matrices.
    protein : str
        Protein to correlate.
    """
    radii = sorted(radius_activities.keys(), key=float)
    n = len(radii)
    if n == 0:
        return _empty_figure("No radius data for correlation")

    fig = make_subplots(rows=1, cols=n, subplot_titles=[f"{r} \u03bcm" for r in radii])

    for i, radius in enumerate(radii, 1):
        mat = radius_activities[radius]
        if protein not in target_activity.index or protein not in mat.index:
            continue
        target = target_activity.loc[protein]
        neighbor = mat.loc[protein]
        common = target.index.intersection(neighbor.index)
        x = target[common].astype(float).values
        y = neighbor[common].astype(float).values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", marker=dict(size=3, color=_PRIMARY, opacity=0.5),
            showlegend=False,
        ), row=1, col=i)

        if len(x) > 2:
            coeffs = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            x_line = np.array([x.min(), x.max()])
            fig.add_trace(go.Scatter(
                x=x_line, y=np.polyval(coeffs, x_line),
                mode="lines", line=dict(color=_ACCENT, width=2),
                showlegend=False,
            ), row=1, col=i)
            fig.add_annotation(
                x=0.05, y=0.95, xref=f"x{i} domain", yref=f"y{i} domain",
                text=f"r={r:.3f}<br>n={len(x)}", showarrow=False,
                font=dict(size=10), bgcolor="white",
            )

    fig.update_layout(
        title=f"{protein} Activity Correlation: Target vs Radius",
        template="plotly_white", height=400,
    )
    return fig


def gene_expression_stats(expression_matrix: pd.DataFrame) -> go.Figure:
    """Scatter: mean expression vs detection rate per gene.

    Parameters
    ----------
    expression_matrix : DataFrame
        Genes x spots (normalized counts).
    """
    mean_expr = expression_matrix.mean(axis=1)
    detection = (expression_matrix > 0).sum(axis=1) / expression_matrix.shape[1] * 100

    fig = go.Figure(go.Scatter(
        x=mean_expr, y=detection, mode="markers",
        marker=dict(size=4, color=_PRIMARY, opacity=0.6),
        text=expression_matrix.index,
        hoverinfo="text+x+y",
    ))
    fig.update_layout(
        title="Gene Expression Distribution",
        xaxis_title="Mean Expression",
        yaxis_title="Detection Rate (%)",
        template="plotly_white",
    )
    return fig


def celltype_expression_boxplot(
    expression_matrix: pd.DataFrame,
    cell_types: pd.Series,
    gene: str,
) -> go.Figure:
    """Boxplot of gene expression by cell type.

    Parameters
    ----------
    expression_matrix : DataFrame
        Genes x spots.
    cell_types : Series
        Cell type per spot.
    gene : str
        Gene to plot.
    """
    if gene not in expression_matrix.index:
        return _empty_figure(f"Gene {gene} not found")
    values = expression_matrix.loc[gene]
    common = values.index.intersection(cell_types.index)
    df = pd.DataFrame({"expression": values[common], "celltype": cell_types[common]}).dropna()

    order = df.groupby("celltype")["expression"].median().sort_values(ascending=False).index
    fig = go.Figure()
    for ct in order:
        subset = df[df["celltype"] == ct]["expression"]
        fig.add_trace(go.Box(y=subset, name=ct, boxmean=True))
    fig.update_layout(
        title=f"{gene} Expression by Cell Type",
        yaxis_title="Expression",
        showlegend=False,
        template="plotly_white",
    )
    return fig


def celltype_distribution(cell_types: pd.Series) -> go.Figure:
    """Bar chart of cell type counts.

    Parameters
    ----------
    cell_types : Series
        Cell type label per spot.
    """
    counts = cell_types.value_counts().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(), y=counts.values,
        marker_color=_PRIMARY,
    ))
    fig.update_layout(
        title="Cell Type Distribution",
        xaxis_title="Cell Type",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


def spatial_density(coordinates: pd.DataFrame) -> go.Figure:
    """2D density contour of spatial distribution.

    Parameters
    ----------
    coordinates : DataFrame
        Must have columns 'x' and 'y'.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coordinates["x"], y=coordinates["y"],
        mode="markers", marker=dict(size=2, color=_PRIMARY, opacity=0.3),
        showlegend=False,
    ))
    fig.add_trace(go.Histogram2dContour(
        x=coordinates["x"], y=coordinates["y"],
        colorscale="Blues", showscale=True,
        contours=dict(showlabels=False),
        opacity=0.4,
    ))
    fig.update_layout(
        title=f"Spatial Distribution ({len(coordinates)} spots)",
        xaxis_title="X", yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Placeholder figure for missing data or errors."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        template="plotly_white",
    )
    return fig
