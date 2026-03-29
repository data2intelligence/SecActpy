"""Inference tab callbacks — upload, run secact_activity, display results."""

import io
import os
import base64
import tempfile

import pandas as pd
from dash import Input, Output, State, no_update


def register_inference_callbacks(app):
    """Register inference tab callbacks."""

    _inference_cache = {}

    def _evict_cache():
        _inference_cache.clear()
        import gc
        gc.collect()

    @app.callback(
        [Output("inference-run-btn", "disabled"),
         Output("inference-data-store", "data")],
        Input("inference-upload", "contents"),
        State("inference-upload", "filename"),
        prevent_initial_call=True,
    )
    def on_upload(contents, filename):
        """Enable run button when file is uploaded."""
        if not contents:
            return True, no_update
        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            ext = filename.rsplit(".", 1)[-1].lower() if filename else ""

            if ext == "h5ad":
                with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
                    f.write(decoded)
                    tmp_path = f.name
                import anndata as ad
                adata = ad.read_h5ad(tmp_path)
                os.remove(tmp_path)
                # Convert to DataFrame (genes x samples)
                if hasattr(adata.X, "toarray"):
                    df = pd.DataFrame(adata.X.toarray().T,
                                      index=adata.var_names, columns=adata.obs_names)
                else:
                    df = pd.DataFrame(adata.X.T,
                                      index=adata.var_names, columns=adata.obs_names)
            elif ext == "csv":
                df = pd.read_csv(io.BytesIO(decoded), index_col=0)
            else:
                df = pd.read_csv(io.BytesIO(decoded), index_col=0, sep="\t")

            _evict_cache()
            _inference_cache["expression"] = df
            return False, {"uploaded": True, "shape": list(df.shape), "filename": filename}
        except Exception as e:
            return True, {"error": str(e)}

    @app.callback(
        [Output("inference-welcome", "style"),
         Output("inference-results-area", "style"),
         Output("inference-results-table", "data"),
         Output("inference-results-table", "columns"),
         Output("inference-status", "children"),
         Output("inference-run-btn", "disabled")],
        Input("inference-run-btn", "n_clicks"),
        [State("inference-input-type", "value"),
         State("inference-data-store", "data")],
        prevent_initial_call=True,
    )
    def run_inference(n_clicks, input_type, data_store):
        """Run SecAct inference on uploaded data."""
        if not data_store or "expression" not in _inference_cache:
            return (no_update,) * 6

        try:
            from secactpy import secact_activity_inference

            expr_df = _inference_cache["expression"]
            is_diff = input_type == "logFC"

            result = secact_activity_inference(
                expr_df,
                is_differential=is_diff,
                verbose=False,
            )

            # Extract z-scores
            zscore_df = result.get("zscore", pd.DataFrame())
            if isinstance(zscore_df, pd.DataFrame) and not zscore_df.empty:
                # Add protein names as column
                display_df = zscore_df.reset_index()
                display_df.columns = ["Protein"] + list(display_df.columns[1:])
                _inference_cache["results"] = display_df

                columns = [{"name": c, "id": c} for c in display_df.columns]
                data = display_df.round(4).to_dict("records")

                return (
                    {"display": "none"},
                    {"display": "block"},
                    data,
                    columns,
                    "Inference complete!",
                    False,  # re-enable button
                )
            else:
                return (no_update, no_update, [], [], "No results returned", False)
        except Exception as e:
            return (no_update, no_update, [], [], f"Error: {e}", False)

    @app.callback(
        Output("inference-download", "data"),
        Input("inference-download-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_results(n_clicks):
        """Download results as CSV."""
        if "results" not in _inference_cache:
            return no_update
        from dash import dcc
        return dcc.send_data_frame(_inference_cache["results"].to_csv, "secact_results.csv", index=False)
