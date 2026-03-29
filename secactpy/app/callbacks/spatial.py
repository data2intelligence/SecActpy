"""Spatial tab callbacks — wired to spatial-gpu I/O and secactpy visualization."""

import os
import base64
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

from secactpy.app.utils import empty_figure


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

    # Single-slot cache: stores only the current dataset. Previous data is evicted
    # on each new upload to prevent unbounded memory growth.
    _data_cache = {}

    def _evict_cache():
        _data_cache.clear()
        import gc
        gc.collect()

    def _load_platform_zip(contents, platform):
        """Extract a platform zip and load via spatial-gpu readers."""
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        extract_dir = Path(tempfile.mkdtemp(prefix=f"{platform}_"))
        zip_path = extract_dir / "upload.zip"
        zip_path.write_bytes(decoded)

        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir / "data")

        data_dir = extract_dir / "data"
        # Find the actual output directory (may be nested)
        subdirs = list(data_dir.iterdir())
        if len(subdirs) == 1 and subdirs[0].is_dir():
            data_dir = subdirs[0]

        try:
            from spatialgpu.io import read_visium, read_cosmx, read_xenium
            readers = {"visium": read_visium, "cosmx": read_cosmx, "xenium": read_xenium}
            return readers[platform](str(data_dir))
        except ImportError:
            # Fallback: try scanpy
            import scanpy as sc
            if platform == "visium":
                return sc.read_visium(str(data_dir))
            raise ImportError(f"spatial-gpu required for {platform} data loading")

    @app.callback(
        [Output("spatial-welcome", "style"),
         Output("spatial-plot-area", "style"),
         Output("spatial-controls", "style"),
         Output("spatial-feature", "options"),
         Output("spatial-feature", "value"),
         Output("spatial-data-store", "data"),
         Output("spatial-inference-btn", "style")],
        [Input("spatial-demo-btn", "n_clicks"),
         Input("spatial-upload", "contents"),
         Input("spatial-visium-btn", "n_clicks"),
         Input("spatial-cosmx-btn", "n_clicks"),
         Input("spatial-xenium-btn", "n_clicks")],
        [State("spatial-upload", "filename"),
         State("spatial-visium-upload", "contents"),
         State("spatial-cosmx-upload", "contents"),
         State("spatial-xenium-upload", "contents")],
        prevent_initial_call=True,
    )
    def load_data(demo_clicks, upload_contents, visium_clicks, cosmx_clicks, xenium_clicks,
                  upload_filename, visium_contents, cosmx_contents, xenium_contents):
        """Load data from demo, file upload, or platform-specific zip."""
        from dash import callback_context
        triggered = callback_context.triggered[0]["prop_id"]

        try:
            adata = None

            if "demo-btn" in triggered:
                try:
                    import scanpy as sc
                    adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
                    adata.var_names_make_unique()
                except Exception:
                    return (no_update,) * 7

            elif "spatial-upload" in triggered and upload_contents:
                content_type, content_string = upload_contents.split(",")
                decoded = base64.b64decode(content_string)
                ext = upload_filename.rsplit(".", 1)[-1].lower() if upload_filename else ""
                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                    f.write(decoded)
                    tmp_path = f.name
                if ext == "h5ad":
                    import anndata as ad
                    adata = ad.read_h5ad(tmp_path)
                    os.remove(tmp_path)
                else:
                    # CSV/TSV — assume genes x spots
                    import anndata as ad
                    sep = "," if ext == "csv" else "\t"
                    df = pd.read_csv(tmp_path, index_col=0, sep=sep)
                    os.remove(tmp_path)
                    adata = ad.AnnData(df.T)  # transpose to spots x genes

            elif "visium-btn" in triggered and visium_contents:
                adata = _load_platform_zip(visium_contents, "visium")

            elif "cosmx-btn" in triggered and cosmx_contents:
                adata = _load_platform_zip(cosmx_contents, "cosmx")

            elif "xenium-btn" in triggered and xenium_contents:
                adata = _load_platform_zip(xenium_contents, "xenium")

            if adata is None:
                return (no_update,) * 7

            _evict_cache()
            _data_cache["current"] = adata

            # Build feature list
            features = []
            for col in adata.obs.columns:
                features.append({"label": f"[obs] {col}", "value": col})
            if hasattr(adata.X, "toarray"):
                variances = np.asarray(adata.X.toarray().var(axis=0))
            else:
                variances = np.asarray(adata.X.var(axis=0)).ravel()
            top_genes = adata.var_names[np.argsort(variances)[-50:]].tolist()
            for g in sorted(top_genes):
                features.append({"label": f"[gene] {g}", "value": g})

            default = features[0]["value"] if features else None

            return (
                {"display": "none"},
                {"display": "block"},
                {"display": "block"},
                features,
                default,
                {"loaded": True, "n_obs": adata.n_obs, "n_vars": adata.n_vars},
                {"display": "block"},
            )
        except Exception:
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
