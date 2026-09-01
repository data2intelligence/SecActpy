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
    "ccc_heatmap",
    "secreted_protein_dotplot",
    "secreted_protein_split_heatmap",
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


def ccc_heatmap(
    ccc: Any,
    *,
    compare_to: Any = None,
    labels: tuple = ("Case", "Control"),
    row_sorted: bool = False,
    column_sorted: bool = False,
    title: str | None = None,
    colorscale: Any = None,
) -> go.Figure:
    """Sender x receiver cell-cell communication count heatmap.

    Python port of R SecAct's ``SecAct.CCC.heatmap``: a heatmap of the number
    of secreted-protein interaction edges from each **sender** cell type (rows)
    to each **receiver** cell type (columns), with the count printed in each
    cell. Rows/columns cover the union of sender and receiver cell types so the
    two axes align.

    Parameters
    ----------
    ccc : DataFrame or dict
        The CCC edge table (``secreted_protein_ccc`` from
        :func:`secactpy.secact_ccc_scrnaseq`) with ``sender`` and ``receiver``
        columns; the full result dict is also accepted.
    row_sorted, column_sorted : bool
        Sort senders / receivers by total interaction count (descending).
    title : str, optional
        Centered plot title.
    colorscale : optional
        Plotly colorscale; default white -> red (interaction counts are >= 0).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    m1 = _ccc_matrix(ccc)
    m2 = _ccc_matrix(compare_to) if compare_to is not None else None
    if m1.empty and (m2 is None or m2.empty):
        return _empty_figure("No cell-cell communication edges to plot")

    # One cell-type universe and ONE colour scale across panels. Scaling each
    # panel to its own maximum would make two very different edge counts render
    # as the same red, which is the opposite of what a contrast is for: here the
    # non-responder arm carries several times the responder arm's edges, and that
    # difference has to be visible rather than normalized away.
    cell_types = sorted(set(m1.index) | set(m1.columns)
                        | (set(m2.index) | set(m2.columns) if m2 is not None else set()))
    m1 = m1.reindex(index=cell_types, columns=cell_types, fill_value=0)
    if m2 is not None:
        m2 = m2.reindex(index=cell_types, columns=cell_types, fill_value=0)

    order_src = m1 if m2 is None else m1.add(m2, fill_value=0)
    if row_sorted:
        idx = order_src.sum(axis=1).sort_values(ascending=False).index
        m1 = m1.loc[idx]
        m2 = None if m2 is None else m2.loc[idx]
    if column_sorted:
        cidx = order_src.sum(axis=0).sort_values(ascending=False).index
        m1 = m1[cidx]
        m2 = None if m2 is None else m2[cidx]

    if colorscale is None:
        colorscale = [[0.0, "#f7f7f7"], [1.0, "#cf3a2e"]]
    zmax = float(max(m1.values.max(), 0 if m2 is None else m2.values.max())) or 1.0

    if m2 is not None:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=list(labels[:2]),
                            horizontal_spacing=0.13, shared_yaxes=True)
        for k, m in enumerate((m1, m2)):
            z = m.values[::-1]
            fig.add_trace(go.Heatmap(
                z=z, x=[str(c) for c in m.columns],
                y=[str(r) for r in m.index[::-1]],
                colorscale=colorscale, zmin=0, zmax=zmax, xgap=1, ygap=1,
                text=z, texttemplate="%{text}", textfont=dict(size=13),
                showscale=(k == 1), colorbar=dict(title="Edges"),
                hovertemplate="sender: %{y}<br>receiver: %{x}<br>edges: %{z}<extra></extra>",
            ), row=1, col=k + 1)
            fig.update_xaxes(title="Receiver", tickangle=45, ticks="", row=1, col=k + 1)
        fig.update_yaxes(title="Sender", ticks="", row=1, col=1)
        fig.update_layout(
            title=dict(text=title or "", x=0.5, xanchor="center"),
            template="plotly_white",
            margin=dict(l=10, r=10, t=70 if title else 40, b=10))
        return fig

    mat = m1

    # reverse rows so the first sender is at the top (matches R's row order)
    z = mat.values[::-1]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(c) for c in mat.columns],
        y=[str(r) for r in mat.index[::-1]],
        colorscale=colorscale, zmin=0,
        xgap=1, ygap=1,
        text=z, texttemplate="%{text}", textfont=dict(size=12),
        colorbar=dict(title="Edges"),
        hovertemplate="sender: %{y}<br>receiver: %{x}<br>edges: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        template="plotly_white",
        xaxis=dict(title="Receiver", side="bottom", tickangle=90, ticks="", constrain="domain"),
        yaxis=dict(title="Sender", ticks="", automargin=True),
        margin=dict(l=10, r=10, t=44 if title else 20, b=10),
    )
    return fig


# diverging (signed activity) and sequential (non-negative spread) colorscales
_DIVERGING = [[0.0, "#2166AC"], [0.5, "#F7F7F7"], [1.0, "#B2182B"]]
_SEQ_SPREAD = [[0.0, "#F7F7F7"], [1.0, "#542788"]]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _sample_colorscale(t, colorscale):
    """Interpolate a plotly colorscale (list of [pos, '#hex']) at t in [0,1] -> 'rgb(...)'."""
    t = float(min(1.0, max(0.0, t)))
    stops = [(float(p), _hex_to_rgb(c)) for p, c in colorscale]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if t <= p1:
            f = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            rgb = [round(a + (b - a) * f) for a, b in zip(c0, c1)]
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    r = stops[-1][1]
    return f"rgb({r[0]},{r[1]},{r[2]})"


def _top_n_rows(mat, top_n):
    """Union of top-`top_n` rows per column, ordered by the column each peaks in."""
    keep, seen = [], set()
    for col in mat.columns:
        for r in mat[col].sort_values(ascending=False).head(top_n).index:
            if r not in seen:
                seen.add(r)
                keep.append(r)
    col_rank = {c: i for i, c in enumerate(mat.columns)}
    peak = mat.loc[keep].idxmax(axis=1)
    keep.sort(key=lambda r: (col_rank[peak[r]], -float(mat.loc[r, peak[r]])))
    return keep


def _resolve_activity_uncertainty(activity, *, sc, se=None, cell_types=None,
                                  spread="sd", min_cells=30):
    """Resolve (intensity, uncertainty, label) for the activity/uncertainty plots.

    ``sc=False`` (cell-type level): ``activity`` is a proteins x cell-types
    activity matrix; uncertainty is the ridge **SE** matrix (``se``, same shape).
    ``sc=True`` (single-cell level): ``activity`` is per-cell (proteins x cells)
    and ``cell_types`` labels each column; intensity = per-cell-type **mean**,
    uncertainty = per-cell-type **spread** (``sd`` / ``cv`` / ``iqr``), with cell
    types under ``min_cells`` dropped. Returns ``(None, None, None)`` when empty.
    """
    activity = pd.DataFrame(activity)
    if not sc:
        mat = activity.dropna(how="all").dropna(axis=1, how="all").astype(float)
        if mat.empty:
            return None, None, None
        unc = (pd.DataFrame(se).reindex(index=mat.index, columns=mat.columns).astype(float)
               if se is not None else None)
        return mat, unc, "SE"
    if cell_types is None:
        raise ValueError("cell_types is required when sc=True (per-cell activity).")
    ct = (cell_types.reindex(activity.columns) if isinstance(cell_types, pd.Series)
          else pd.Series(list(cell_types), index=activity.columns))
    counts = ct.value_counts()
    keep = sorted([c for c in counts.index if counts[c] >= min_cells], key=str.lower)
    if not keep:
        return None, None, None
    mean_df, unc_df = {}, {}
    for c in keep:
        block = activity[ct.index[ct == c]].astype(float)
        m = block.mean(axis=1)
        mean_df[c] = m
        if spread == "iqr":
            s = block.quantile(0.75, axis=1) - block.quantile(0.25, axis=1)
        else:
            s = block.std(axis=1, ddof=1)
            if spread == "cv":
                s = s / m.abs().replace(0, np.nan)
        unc_df[c] = s
    return (pd.DataFrame(mean_df).reindex(columns=keep),
            pd.DataFrame(unc_df).reindex(columns=keep),
            f"per-cell {spread.upper()}")


def secreted_protein_dotplot(
    activity: pd.DataFrame,
    *,
    sc: bool = False,
    se: Optional[pd.DataFrame] = None,
    cell_types: Any = None,
    spread: str = "sd",
    min_cells: int = 30,
    invert_size: bool = False,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    colorscale: Any = None,
    max_marker: float = 26.0,
    min_marker: float = 4.0,
) -> go.Figure:
    """Dot heatmap of secreted-protein activity: color = activity, size = uncertainty.

    Scanpy-style dotplot. **Color** = signed activity (diverging scale). **Size**
    = the uncertainty channel, chosen by ``sc``:

    - ``sc=False`` (cell-type level): size = the ridge **SE** (pass ``se``, a
      proteins x cell-types matrix aligned to ``activity``).
    - ``sc=True`` (single-cell level): ``activity`` is per-cell (proteins x cells)
      and ``cell_types`` labels each column; color = per-cell-type mean, size =
      per-cell **spread** (``spread``), cell types under ``min_cells`` dropped.

    By default a larger dot means *more* uncertainty; set ``invert_size=True`` to
    map size to precision (1/uncertainty), so larger = more confident.
    ``top_n`` keeps the union of the top-``top_n`` proteins per cell type.
    """
    colorscale = colorscale or _DIVERGING
    mat, unc, unc_label = _resolve_activity_uncertainty(
        activity, sc=sc, se=se, cell_types=cell_types, spread=spread, min_cells=min_cells)
    if mat is None or mat.empty:
        return _empty_figure("No activity values to plot")
    if top_n:
        rows = _top_n_rows(mat.fillna(-np.inf), top_n)
        mat = mat.loc[rows]
        if unc is not None:
            unc = unc.loc[rows]

    proteins = list(mat.index)
    cell_types_ax = list(mat.columns)
    ypos = {p: len(proteins) - 1 - i for i, p in enumerate(proteins)}
    size_desc = None if unc is None else (f"precision (1/{unc_label})" if invert_size else unc_label)

    def _to_size(v):
        if unc is None or not np.isfinite(v):
            return np.nan
        return (1.0 / v) if (invert_size and v != 0) else float(v)

    if unc is not None:
        sv = np.array([_to_size(v) for v in unc.values.ravel()], dtype=float)
        finite = sv[np.isfinite(sv)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

    xs, ys, colors, sizes, hover = [], [], [], [], []
    for p in proteins:
        for c in cell_types_ax:
            xs.append(c); ys.append(ypos[p]); colors.append(float(mat.loc[p, c]))
            if unc is not None:
                sval = _to_size(unc.loc[p, c])
                px = ((min_marker + max_marker) / 2 if (not np.isfinite(sval) or hi == lo)
                      else min_marker + (max_marker - min_marker) * (sval - lo) / (hi - lo))
                sizes.append(px)
                hover.append(f"{p} · {c}<br>activity: {mat.loc[p,c]:.2f}<br>{unc_label}: {float(unc.loc[p,c]):.3g}")
            else:
                sizes.append((min_marker + max_marker) / 2)
                hover.append(f"{p} · {c}<br>activity: {mat.loc[p,c]:.2f}")

    cmax = float(np.nanmax(np.abs(mat.values))) or 1.0
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers", showlegend=False,
        marker=dict(color=colors, colorscale=colorscale, cmid=0, cmin=-cmax, cmax=cmax,
                    size=sizes, line=dict(width=0.5, color="rgba(80,80,80,.5)"),
                    colorbar=dict(title="activity")),
        text=hover, hovertemplate="%{text}<extra></extra>",
    ))
    if unc is not None:
        for frac, lab in [(0.15, "low"), (0.55, "mid"), (1.0, "high")]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=f"{size_desc}: {lab}",
                marker=dict(size=min_marker + (max_marker - min_marker) * frac,
                            color="rgba(120,120,120,.6)")))
    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        template="plotly_white",
        xaxis=dict(title=None, tickangle=90, side="bottom",
                   categoryorder="array", categoryarray=cell_types_ax),
        yaxis=dict(title=None, tickmode="array",
                   tickvals=[ypos[p] for p in proteins], ticktext=proteins,
                   range=[-0.7, len(proteins) - 0.3]),
        margin=dict(l=10, r=10, t=44 if title else 20, b=10),
        legend=dict(itemsizing="trace", y=1, x=1.02),
    )
    return fig


def secreted_protein_split_heatmap(
    activity: pd.DataFrame,
    *,
    sc: bool = False,
    se: Optional[pd.DataFrame] = None,
    cell_types: Any = None,
    spread: str = "sd",
    top_n: Optional[int] = None,
    min_cells: int = 30,
    title: Optional[str] = None,
    intensity_colorscale: Any = None,
    spread_colorscale: Any = None,
) -> go.Figure:
    """Split-cell heatmap: bottom-right = activity, top-left = uncertainty.

    Each cell is split along the ``/`` diagonal: the **bottom-right** triangle
    shows the secreted-protein activity (diverging scale, signed) and the
    **top-left** triangle shows its uncertainty (sequential scale). The
    uncertainty channel is chosen by ``sc``:

    - ``sc=False`` (cell-type level): ``activity`` is a proteins x cell-types
      matrix; the top-left triangle is the ridge **SE** (pass ``se``, same shape).
    - ``sc=True`` (single-cell level): ``activity`` is per-cell (proteins x cells)
      and ``cell_types`` labels each column; bottom-right = per-cell-type mean,
      top-left = per-cell **spread** (``spread`` = ``sd``/``cv``/``iqr``), with
      cell types under ``min_cells`` dropped.

    Makes within-cell-type heterogeneity (or estimation uncertainty) visible
    alongside the activity level. Best at modest size (top-N proteins x few cell
    types); use ``top_n``.
    """
    intensity_colorscale = intensity_colorscale or _DIVERGING
    spread_colorscale = spread_colorscale or _SEQ_SPREAD
    mean_mat, unc_mat, unc_label = _resolve_activity_uncertainty(
        activity, sc=sc, se=se, cell_types=cell_types, spread=spread, min_cells=min_cells)
    if mean_mat is None or mean_mat.empty:
        return _empty_figure(f"No cell type has >= {min_cells} cells" if sc
                             else "No activity values to plot")
    if unc_mat is None:
        raise ValueError("se is required when sc=False (the top-left triangle "
                         "needs an uncertainty matrix).")
    if top_n:
        rows = _top_n_rows(mean_mat.fillna(-np.inf), top_n)
        mean_mat, unc_mat = mean_mat.loc[rows], unc_mat.loc[rows]

    keep_ct = list(mean_mat.columns)
    proteins = list(mean_mat.index)
    n_p, n_c = len(proteins), len(keep_ct)
    if n_p == 0:
        return _empty_figure("No proteins to plot")

    vmax = float(np.nanmax(np.abs(mean_mat.values))) or 1.0
    smax = float(np.nanmax(unc_mat.values)) or 1.0

    # Collect the triangles and assign them ONCE. `fig.add_shape` appends to an
    # immutable tuple and revalidates every existing shape on each call, so a
    # per-triangle loop is quadratic in the number of cells: measured at 210 / 520
    # / 1,600 shapes it took 4.0s / 23.5s / 302s, which extrapolates to ~31 min for
    # a 1,170 x 51 panel -- and that is exactly where a real run stalled. One
    # assignment validates the list a single time.
    shapes = []
    mean_v = mean_mat.reindex(index=proteins, columns=keep_ct).values
    unc_v = unc_mat.reindex(index=proteins, columns=keep_ct).values
    for ri in range(len(proteins)):
        y = n_p - 1 - ri  # first protein on top
        for ci in range(len(keep_ct)):
            x = ci
            mv, sv = mean_v[ri, ci], unc_v[ri, ci]
            if np.isfinite(mv):  # bottom-right triangle = activity (diverging, centered 0)
                t = 0.5 + 0.5 * (float(mv) / vmax)
                shapes.append(dict(type="path",
                    path=f"M {x-0.5},{y-0.5} L {x+0.5},{y-0.5} L {x+0.5},{y+0.5} Z",
                    fillcolor=_sample_colorscale(t, intensity_colorscale),
                    line=dict(width=0.4, color="white")))
            if np.isfinite(sv):  # top-left triangle = uncertainty (sequential)
                shapes.append(dict(type="path",
                    path=f"M {x-0.5},{y-0.5} L {x-0.5},{y+0.5} L {x+0.5},{y+0.5} Z",
                    fillcolor=_sample_colorscale(float(sv) / smax, spread_colorscale),
                    line=dict(width=0.4, color="white")))
    fig = go.Figure()
    fig.update_layout(shapes=shapes)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False,
        marker=dict(colorscale=intensity_colorscale, cmin=-vmax, cmax=vmax, cmid=0,
                    color=[0], showscale=True,
                    colorbar=dict(title="activity", x=1.02, len=0.45, y=0.78))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False,
        marker=dict(colorscale=spread_colorscale, cmin=0, cmax=smax,
                    color=[0], showscale=True,
                    colorbar=dict(title=unc_label, x=1.02, len=0.45, y=0.25))))

    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        template="plotly_white",
        xaxis=dict(tickmode="array", tickvals=list(range(n_c)), ticktext=keep_ct,
                   tickangle=90, range=[-0.5, n_c - 0.5], constrain="domain",
                   zeroline=False, showgrid=False),
        yaxis=dict(tickmode="array", tickvals=list(range(n_p)),
                   ticktext=list(reversed(proteins)), range=[-0.5, n_p - 0.5],
                   zeroline=False, showgrid=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=46 if title else 22, b=10),
        annotations=[dict(text=f"cells: top-left = {unc_label}, bottom-right = activity",
                          xref="paper", yref="paper", x=0, y=1.04, showarrow=False,
                          font=dict(size=10, color="#6C7883"))],
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


def _ccc_matrix(ccc: Any) -> pd.DataFrame:
    """Sender x receiver edge-count matrix on the union of cell types."""
    if isinstance(ccc, dict):
        ccc = ccc.get("secreted_protein_ccc")
    cols = getattr(ccc, "columns", [])
    if ccc is None or len(ccc) == 0 or not {"sender", "receiver"}.issubset(set(cols)):
        return pd.DataFrame()
    mat = pd.crosstab(ccc["sender"], ccc["receiver"])
    types = sorted(set(mat.index) | set(mat.columns))
    return mat.reindex(index=types, columns=types, fill_value=0)


def _arc(t0, t1, r, n=40):
    a = np.linspace(t0, t1, n)
    return r * np.cos(a), r * np.sin(a)


def _ribbon(t0, t1, s0, s1, r, n=40):
    """Filled chord between arc spans [t0,t1] and [s0,s1].

    Both flanks are quadratic Beziers pulled toward the centre, which is what
    makes a chord read as a connection rather than as a polygon: a straight
    flank between two arcs on a circle looks like a wedge of the disc.
    """
    def bez(a, b, m=n):
        p0 = np.array([r * np.cos(a), r * np.sin(a)])
        p2 = np.array([r * np.cos(b), r * np.sin(b)])
        u = np.linspace(0, 1, m)[:, None]
        # control point at the origin: pull scales with angular separation, so
        # near neighbours keep a shallow chord and opposite pairs bow deeply
        p1 = np.zeros(2) * 0.0
        pts = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2
        return pts[:, 0], pts[:, 1]

    ax, ay = _arc(t0, t1, r)
    bx, by = bez(t1, s0)
    cx, cy = _arc(s0, s1, r)
    dx, dy = bez(s1, t0)
    return np.concatenate([ax, bx, cx, dx]), np.concatenate([ay, by, cy, dy])


_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
            "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#E377C2", "#7F7F7F"]


def _circle_panel(mat, geom, colors, sender, receiver, gap=0.02, r=1.0):
    """Traces for one chord diagram; returns (traces, annotations).

    ``geom`` sizes the arcs and ``mat`` supplies the chords. They are separate so
    a multi-panel contrast can share one geometry: sizing each panel from its own
    totals would rescale the arcs independently, and a cell type with fewer edges
    would render as a different-sized node rather than as the same node sending
    less.
    """
    types = list(geom.index)
    total = (geom.sum(axis=1) + geom.sum(axis=0)).reindex(types).fillna(0).values
    if total.sum() <= 0:
        return [], []
    span = (2 * np.pi - gap * len(types)) * total / total.sum()
    start = np.cumsum(np.r_[0.0, span[:-1]] + gap) - gap
    pos = {t: (start[i], start[i] + span[i]) for i, t in enumerate(types)}

    traces = []
    for i, t in enumerate(types):                      # node arcs
        x, y = _arc(*pos[t], r=r)
        xo, yo = _arc(pos[t][1], pos[t][0], r=r * 1.06)
        traces.append(go.Scatter(
            x=np.r_[x, xo], y=np.r_[y, yo], fill="toself", mode="lines",
            line=dict(width=0), fillcolor=colors[t], hoverinfo="text",
            text=f"{t}<br>out {int(mat.loc[t].sum())} / in {int(mat[t].sum())}",
            showlegend=False))

    # Each sender's outgoing edges consume its arc left to right, and each
    # receiver's incoming edges consume its own arc, so a chord lands on a
    # distinct slice at both ends rather than every chord stacking on the arc
    # midpoint -- which is what makes the widths readable as counts.
    out_cur = {t: pos[t][0] for t in types}
    in_cur = {t: pos[t][0] for t in types}
    out_tot = geom.sum(axis=1)
    in_tot = geom.sum(axis=0)
    for s in types:
        for rc in types:
            v = float(mat.loc[s, rc])
            if v <= 0:
                continue
            ws = span[types.index(s)] * v / max(out_tot[s] + in_tot[s], 1e-9)
            wr = span[types.index(rc)] * v / max(out_tot[rc] + in_tot[rc], 1e-9)
            t0, t1 = out_cur[s], out_cur[s] + ws
            s0, s1 = in_cur[rc], in_cur[rc] + wr
            out_cur[s] = t1
            in_cur[rc] = s1
            visible = ((sender is None or s in sender)
                       and (receiver is None or rc in receiver))
            x, y = _ribbon(t0, t1, s0, s1, r * 0.98)
            traces.append(go.Scatter(
                x=x, y=y, fill="toself", mode="lines", line=dict(width=0),
                fillcolor=colors[s] if visible else "rgba(0,0,0,0)",
                opacity=0.55 if visible else 0.0, hoverinfo="text",
                text=f"{s} -> {rc}: {int(v)} edges", showlegend=False))

    ann = []
    for t in types:
        a = np.mean(pos[t])
        ann.append(dict(x=r * 1.16 * np.cos(a), y=r * 1.16 * np.sin(a), text=t,
                        showarrow=False, font=dict(size=11),
                        xanchor="left" if np.cos(a) >= 0 else "right"))
    return traces, ann


def ccc_circle(
    ccc: Any,
    *,
    compare_to: Any = None,
    labels: tuple = ("Case", "Control"),
    colors: Any = None,
    sender: Any = None,
    receiver: Any = None,
    title: str | None = None,
) -> go.Figure:
    """Chord diagram of sender -> receiver secreted-protein communication.

    Python port of R SecAct's ``SecAct.CCC.circle`` (``circlize::chordDiagram``).
    Cell types sit on a circle with arc length proportional to their total
    interaction count; each chord is one sender -> receiver pair with width
    proportional to the number of significant secreted-protein edges, coloured by
    the SENDER so direction is readable without arrowheads.

    Parameters
    ----------
    ccc : DataFrame or dict
        Edge table (``secreted_protein_ccc``) or the full result dict from
        :func:`secactpy.secact_ccc_scrnaseq`.
    compare_to : DataFrame or dict, optional
        A second edge table drawn beside the first as a **contrast** — e.g.
        responder vs non-responder. Both panels share one cell-type ordering,
        one colour map, and one arc geometry derived from the COMBINED totals,
        so the two circles are visually comparable; drawing each panel on its own
        geometry would rescale the arcs independently and make a cell type that
        merely has fewer edges look like a different cell type.
    labels : tuple
        Panel titles when ``compare_to`` is given.
    colors : dict or list, optional
        Cell-type colours. A dict maps names to colours; a list is cycled.
    sender, receiver : str or iterable, optional
        Restrict which chords are drawn, as R's ``sender``/``receiver`` do:
        the arcs stay in place and non-matching chords become transparent, so
        the filtered view keeps the same geometry as the unfiltered one.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from plotly.subplots import make_subplots

    m1 = _ccc_matrix(ccc)
    m2 = _ccc_matrix(compare_to) if compare_to is not None else None
    if m1.empty and (m2 is None or m2.empty):
        return _empty_figure("No cell-cell communication edges to plot")

    # One shared cell-type universe and one shared geometry across panels.
    types = sorted(set(m1.index) | (set(m2.index) if m2 is not None else set()))
    m1 = m1.reindex(index=types, columns=types, fill_value=0) if not m1.empty \
        else pd.DataFrame(0, index=types, columns=types)
    if m2 is not None:
        m2 = m2.reindex(index=types, columns=types, fill_value=0) if not m2.empty \
            else pd.DataFrame(0, index=types, columns=types)

    if isinstance(colors, dict):
        cmap = {t: colors.get(t, _PALETTE[i % len(_PALETTE)]) for i, t in enumerate(types)}
    else:
        pal = list(colors) if colors is not None else _PALETTE
        cmap = {t: pal[i % len(pal)] for i, t in enumerate(types)}

    if isinstance(sender, str):
        sender = [sender]
    if isinstance(receiver, str):
        receiver = [receiver]

    mats = [m1] if m2 is None else [m1, m2]
    # Shared arc geometry: sized by the COMBINED totals so a chord of the same
    # width means the same count in either panel.
    geom = sum(mats)
    fig = make_subplots(rows=1, cols=len(mats),
                        subplot_titles=list(labels[:len(mats)]) if m2 is not None else None,
                        horizontal_spacing=0.06)
    for k, m in enumerate(mats):
        traces, ann = _circle_panel(m, geom, cmap, sender, receiver)
        for tr in traces:
            fig.add_trace(tr, row=1, col=k + 1)
        for a in ann:
            fig.add_annotation(a, row=1, col=k + 1)

    for k in range(len(mats)):
        fig.update_xaxes(visible=False, range=[-1.45, 1.45], row=1, col=k + 1)
        fig.update_yaxes(visible=False, range=[-1.45, 1.45],
                         scaleanchor=("x" if k == 0 else f"x{k + 1}"),
                         row=1, col=k + 1)
    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        template="plotly_white", showlegend=False,
        margin=dict(l=10, r=10, t=60 if title else 34, b=10))
    return fig
