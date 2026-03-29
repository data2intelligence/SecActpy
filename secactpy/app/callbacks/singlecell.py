"""Single-cell tab callbacks — upload, run SC inference, visualize."""

import base64
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

from secactpy.app.utils import empty_figure


def register_singlecell_callbacks(app):
    """Register single-cell tab callbacks."""

    _sc_cache = {}

    def _evict_cache():
        _sc_cache.clear()
        import gc
        gc.collect()

    @app.callback(
        [Output("sc-run-btn", "disabled"),
         Output("sc-data-store", "data")],
        Input("sc-upload", "contents"),
        State("sc-upload", "filename"),
        prevent_initial_call=True,
    )
    def on_upload(contents, filename):
        if not contents:
            return True, no_update
        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
                f.write(decoded)
                tmp_path = f.name
            import anndata as ad
            adata = ad.read_h5ad(tmp_path)
            os.remove(tmp_path)
            _evict_cache()
            _sc_cache["adata"] = adata
            return False, {"uploaded": True, "n_obs": adata.n_obs, "n_vars": adata.n_vars}
        except Exception as e:
            return True, {"error": str(e)}

    @app.callback(
        [Output("sc-welcome", "style"),
         Output("sc-results-area", "style"),
         Output("sc-controls", "style"),
         Output("sc-protein", "options"),
         Output("sc-protein", "value"),
         Output("sc-results-table", "data"),
         Output("sc-results-table", "columns"),
         Output("sc-status", "children")],
        Input("sc-run-btn", "n_clicks"),
        [State("sc-celltype-col", "value"),
         State("sc-data-store", "data")],
        prevent_initial_call=True,
    )
    def run_sc_inference(n_clicks, cell_type_col, data_store):
        if not data_store or "adata" not in _sc_cache:
            return (no_update,) * 8

        try:
            from secactpy import secact_activity_inference_scrnaseq

            adata = _sc_cache["adata"]
            result = secact_activity_inference_scrnaseq(
                adata,
                cell_type_col=cell_type_col,
                is_single_cell_level=False,
                verbose=False,
            )

            zscore_df = result.get("zscore", pd.DataFrame())
            if isinstance(zscore_df, pd.DataFrame) and not zscore_df.empty:
                _sc_cache["zscore"] = zscore_df
                proteins = zscore_df.index.tolist()

                display_df = zscore_df.reset_index()
                display_df.columns = ["Protein"] + list(display_df.columns[1:])
                columns = [{"name": c, "id": c} for c in display_df.columns]
                data = display_df.round(4).to_dict("records")

                return (
                    {"display": "none"},
                    {"display": "block"},
                    {"display": "block"},
                    [{"label": p, "value": p} for p in proteins],
                    proteins[0],
                    data,
                    columns,
                    "Inference complete!",
                )
            return (no_update,) * 7 + ("No results returned",)
        except Exception as e:
            return (no_update,) * 7 + (f"Error: {e}",)

    @app.callback(
        Output("sc-plot", "figure"),
        [Input("sc-protein", "value"),
         Input("sc-plot-type", "value")],
        prevent_initial_call=True,
    )
    def update_sc_plot(protein, plot_type):
        if not protein or "zscore" not in _sc_cache:
            return empty_figure("Run inference first")

        try:
            zscore_df = _sc_cache["zscore"]

            if plot_type == "bar" and protein in zscore_df.index:
                values = zscore_df.loc[protein].sort_values(ascending=False)
                fig = go.Figure(go.Bar(
                    x=values.index.tolist(), y=values.values,
                    marker_color="#3498db",
                ))
                fig.update_layout(
                    title=f"{protein} Activity by Cell Type",
                    xaxis_title="Cell Type", yaxis_title="Activity (z-score)",
                    template="plotly_white",
                )
                return fig

            elif plot_type == "heatmap":
                fig = go.Figure(go.Heatmap(
                    z=zscore_df.values,
                    x=zscore_df.columns.tolist(),
                    y=zscore_df.index.tolist(),
                    colorscale="RdBu_r",
                    colorbar=dict(title="z-score"),
                ))
                fig.update_layout(
                    title="SecAct Activity Heatmap",
                    template="plotly_white",
                    height=max(400, len(zscore_df) * 15),
                )
                return fig

            elif plot_type == "celltype":
                if "adata" in _sc_cache:
                    adata = _sc_cache["adata"]
                    from secactpy.visualization import celltype_distribution
                    col = next((c for c in adata.obs.columns
                                if "cell" in c.lower() and "type" in c.lower()), None)
                    if col:
                        return celltype_distribution(adata.obs[col])
                return empty_figure("No cell type data available")

            return empty_figure(f"Unknown plot type: {plot_type}")
        except Exception as e:
            return empty_figure(f"Plot error: {e}")
