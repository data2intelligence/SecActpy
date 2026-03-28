"""Spatial tab callbacks — wired to spatial-gpu I/O and secactpy visualization."""

import io
import base64
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

from secactpy.app.utils import empty_figure
from secactpy.app.config import UI_COLORS


def _adata_to_spatial_figure(adata, color_key, spot_size=5):
    """Create a Plotly spatial scatter from AnnData."""
    if "spatial" not in adata.obsm:
        return empty_figure("No spatial coordinates in data")

    coords = adata.obsm["spatial"]
    x, y = coords[:, 0], coords[:, 1]

    # Get color values
    if color_key in adata.obs.columns:
        color_vals = adata.obs[color_key]
        if pd.api.types.is_numeric_dtype(color_vals):
            fig = go.Figure(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(size=spot_size, color=color_vals, colorscale="RdBu_r",
                            colorbar=dict(title=color_key), showscale=True),
                text=adata.obs_names, hoverinfo="text",
            ))
        else:
            fig = go.Figure()
            for cat in color_vals.unique():
                mask = color_vals == cat
                fig.add_trace(go.Scatter(
                    x=x[mask], y=y[mask], mode="markers", name=str(cat),
                    marker=dict(size=spot_size),
                ))
    elif color_key in adata.var_names:
        gene_idx = list(adata.var_names).index(color_key)
        values = adata.X[:, gene_idx]
        if hasattr(values, "toarray"):
            values = values.toarray().ravel()
        else:
            values = np.asarray(values).ravel()
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=spot_size, color=values, colorscale="Viridis",
                        colorbar=dict(title=color_key), showscale=True),
            text=adata.obs_names, hoverinfo="text",
        ))
    else:
        return empty_figure(f"Feature '{color_key}' not found")

    fig.update_layout(
        title=color_key,
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def register_spatial_callbacks(app):
    """Register all spatial tab callbacks."""

    # Store the AnnData object in a module-level dict (Dash stores can't hold AnnData)
    _data_cache = {}

    @app.callback(
        [Output("spatial-welcome", "style"),
         Output("spatial-plot-area", "style"),
         Output("spatial-controls", "style"),
         Output("spatial-feature", "options"),
         Output("spatial-feature", "value"),
         Output("spatial-data-store", "data"),
         Output("spatial-inference-btn", "style")],
        [Input("spatial-demo-btn", "n_clicks"),
         Input("spatial-upload", "contents")],
        [State("spatial-upload", "filename")],
        prevent_initial_call=True,
    )
    def load_data(demo_clicks, upload_contents, upload_filename):
        """Load data from demo or upload."""
        from dash import callback_context
        triggered = callback_context.triggered[0]["prop_id"]

        try:
            adata = None

            if "demo-btn" in triggered:
                # Try spatial-gpu demo data, fall back to scanpy
                try:
                    from spatialgpu.io import read_visium
                    import spatialgpu
                    demo_path = Path(spatialgpu.__file__).parent / "data"
                    # If spatial-gpu has no bundled Visium, use scanpy
                    raise FileNotFoundError("Use scanpy fallback")
                except (ImportError, FileNotFoundError):
                    import scanpy as sc
                    adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
                    adata.var_names_make_unique()

            elif "upload" in triggered and upload_contents:
                content_type, content_string = upload_contents.split(",")
                decoded = base64.b64decode(content_string)
                with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
                    f.write(decoded)
                    tmp_path = f.name
                import anndata as ad
                adata = ad.read_h5ad(tmp_path)

            if adata is None:
                return (no_update,) * 7

            _data_cache["current"] = adata

            # Build feature list from obs columns and top variable genes
            features = []
            for col in adata.obs.columns:
                features.append({"label": f"[obs] {col}", "value": col})
            # Add top genes by variance
            if hasattr(adata.X, "toarray"):
                variances = np.asarray(adata.X.toarray().var(axis=0))
            else:
                variances = np.asarray(adata.X.var(axis=0)).ravel()
            top_genes = adata.var_names[np.argsort(variances)[-50:]].tolist()
            for g in sorted(top_genes):
                features.append({"label": f"[gene] {g}", "value": g})

            default = features[0]["value"] if features else None

            return (
                {"display": "none"},       # hide welcome
                {"display": "block"},      # show plot area
                {"display": "block"},      # show controls
                features,
                default,
                {"loaded": True, "n_obs": adata.n_obs, "n_vars": adata.n_vars},
                {"display": "block"},      # show inference button
            )
        except Exception as e:
            return (no_update,) * 7

    @app.callback(
        Output("spatial-plot", "figure"),
        [Input("spatial-feature", "value"),
         Input("spatial-type", "value"),
         Input("spatial-pointsize", "value")],
        State("spatial-data-store", "data"),
        prevent_initial_call=True,
    )
    def update_spatial_plot(feature, spatial_type, point_size, data_store):
        """Render spatial plot."""
        if not data_store or not feature or "current" not in _data_cache:
            return empty_figure("No data loaded")
        try:
            adata = _data_cache["current"]
            return _adata_to_spatial_figure(adata, feature, spot_size=point_size)
        except Exception as e:
            return empty_figure(f"Error: {e}")

    @app.callback(
        Output("spatial-stats-plot", "figure"),
        [Input("spatial-feature", "value"),
         Input("spatial-type", "value")],
        State("spatial-data-store", "data"),
        prevent_initial_call=True,
    )
    def update_stats_plot(feature, spatial_type, data_store):
        """Render statistics plot."""
        if not data_store or not feature or "current" not in _data_cache:
            return empty_figure("No data loaded")
        try:
            adata = _data_cache["current"]
            from secactpy.visualization import gene_expression_stats, celltype_distribution

            if spatial_type == "expression" and feature in adata.var_names:
                # Gene expression stats for all genes
                if hasattr(adata.X, "toarray"):
                    expr_df = pd.DataFrame(adata.X.toarray().T,
                                            index=adata.var_names, columns=adata.obs_names)
                else:
                    expr_df = pd.DataFrame(adata.X.T,
                                            index=adata.var_names, columns=adata.obs_names)
                return gene_expression_stats(expr_df)
            elif feature in adata.obs.columns and not pd.api.types.is_numeric_dtype(adata.obs[feature]):
                return celltype_distribution(adata.obs[feature])
            else:
                return empty_figure("Select a categorical feature for distribution or gene for expression stats")
        except Exception as e:
            return empty_figure(f"Stats error: {e}")
