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
    "activity_change_bar",
    "risk_lollipop",
    "secreted_protein_heatmap",
]

_PRIMARY = "#3498db"
_ACCENT = "#e74c3c"
# SecAct heatmap gradient (low -> high activity), matching R SecAct.heatmap.plot
_SECACT_HEATMAP_COLORS = ["#03c383", "#aad962", "#fbbf45", "#ef6a32"]


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
            xref_str = "x domain" if i == 1 else f"x{i} domain"
            yref_str = "y domain" if i == 1 else f"y{i} domain"
            fig.add_annotation(
                x=0.05, y=0.95, xref=xref_str, yref=yref_str,
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


def activity_change_bar(
    zscore: pd.Series,
    n_top: int = 15,
    title: str = "Activity Change",
) -> go.Figure:
    """Bar plot of top up- and down-regulated secreted proteins.

    Parameters
    ----------
    zscore : Series
        Activity change z-scores, indexed by protein name.
    n_top : int
        Number of top proteins to show from each direction.
    title : str
        Plot title.
    """
    sorted_z = zscore.sort_values()
    top_down = sorted_z.head(n_top)
    top_up = sorted_z.tail(n_top)
    selected = pd.concat([top_down, top_up])

    colors = [_ACCENT if v < 0 else _PRIMARY for v in selected.values]
    fig = go.Figure(go.Bar(
        x=selected.index.tolist(), y=selected.values,
        marker_color=colors,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Secreted Protein",
        yaxis_title="Activity Change (z-score)",
        template="plotly_white",
    )
    return fig


def risk_lollipop(
    risk_scores: pd.Series,
    n_top: int = 15,
    title: str = "Risk Score",
) -> go.Figure:
    """Lollipop plot of top high/low risk secreted proteins.

    Parameters
    ----------
    risk_scores : Series
        Risk z-scores from Cox regression, indexed by protein.
    n_top : int
        Number of proteins from each tail.
    title : str
        Plot title.
    """
    sorted_r = risk_scores.sort_values()
    selected = pd.concat([sorted_r.head(n_top), sorted_r.tail(n_top)])

    fig = go.Figure()
    colors = [_ACCENT if v > 0 else _PRIMARY for v in selected.values]
    fig.add_trace(go.Scatter(
        x=selected.values, y=selected.index.tolist(),
        mode="markers", marker=dict(size=10, color=colors),
        showlegend=False,
    ))
    for protein, val in selected.items():
        fig.add_shape(type="line", x0=0, x1=val,
                      y0=protein, y1=protein,
                      line=dict(color="gray", width=1))
    fig.update_layout(
        title=title,
        xaxis_title="Risk Score (z-score)",
        template="plotly_white",
        height=max(400, len(selected) * 20),
    )
    return fig


def secreted_protein_heatmap(
    activity: pd.DataFrame,
    *,
    top_n: int | None = None,
    title: str | None = None,
    colors: list[str] | None = None,
) -> go.Figure:
    """Secreted-protein activity heatmap per cell type.

    Python port of R SecAct's ``SecAct.heatmap.plot``: a tile heatmap with
    secreted proteins on the rows, cell types on the columns, and per-cell-type
    activity z-scores as the fill, on SecAct's green -> lime -> amber -> orange
    gradient. Rows read top-to-bottom in the matrix's order (first row on top),
    x labels are rotated 90 degrees, and tiles are separated by thin white gaps
    -- the same look as the R ``geom_tile`` version.

    The typical input is the ``zscore`` matrix returned by
    :func:`secactpy.secact_activity_inference_scrnaseq` (proteins x cell types).

    Parameters
    ----------
    activity : DataFrame
        Activity matrix, proteins (rows) x cell types (columns). Values are the
        activity z-scores; positive = high activity, negative = low.
    top_n : int, optional
        If given, keep only the union of the top-``top_n`` most active proteins
        in each cell type -- the vignette's pre-selection (top 10 per cell
        state) -- and order the surviving rows by the cell type they peak in.
        Default ``None`` plots the matrix exactly as given, like the R function.
    title : str, optional
        Centered plot title.
    colors : list of str, optional
        Gradient colors, low -> high. Default is SecAct's 4-stop palette.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> res = secactpy.secact_activity_inference_scrnaseq(adata, cell_type_col="Annotation")
    >>> fig = secreted_protein_heatmap(res["zscore"], top_n=10)
    >>> fig.show()
    """
    colors = list(colors) if colors else _SECACT_HEATMAP_COLORS
    if not isinstance(activity, pd.DataFrame):
        activity = pd.DataFrame(activity)

    mat = activity.dropna(how="all").dropna(axis=1, how="all")
    if mat.empty:
        return _empty_figure("No activity values to plot")
    mat = mat.astype(float)

    if top_n is not None and top_n > 0 and mat.shape[0] > 1:
        keep, seen = [], set()
        for ct in mat.columns:
            for protein in mat[ct].sort_values(ascending=False).head(top_n).index:
                if protein not in seen:
                    seen.add(protein)
                    keep.append(protein)
        # order rows by the cell type each protein peaks in (column order),
        # then by descending peak activity -> a readable block structure
        col_rank = {c: i for i, c in enumerate(mat.columns)}
        peak_ct = mat.loc[keep].idxmax(axis=1)
        keep.sort(key=lambda p: (col_rank[peak_ct[p]], -float(mat.loc[p, peak_ct[p]])))
        mat = mat.loc[keep]

    # even-stop colorscale from the gradient colors == R scale_fill_gradientn
    if len(colors) == 1:
        colorscale = [[0.0, colors[0]], [1.0, colors[0]]]
    else:
        colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]

    # Plotly draws z[0] at the bottom; R shows the first matrix row on top, so
    # reverse rows to match (first protein ends up at the top).
    fig = go.Figure(go.Heatmap(
        z=mat.values[::-1],
        x=[str(c) for c in mat.columns],
        y=[str(r) for r in mat.index[::-1]],
        colorscale=colorscale,
        xgap=1, ygap=1,
        colorbar=dict(title="Activity"),
        hovertemplate="protein: %{y}<br>cell type: %{x}<br>activity: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        template="plotly_white",
        xaxis=dict(side="bottom", tickangle=90, title=None, ticks="", constrain="domain"),
        yaxis=dict(title=None, ticks="", automargin=True),
        margin=dict(l=10, r=10, t=44 if title else 20, b=10),
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
