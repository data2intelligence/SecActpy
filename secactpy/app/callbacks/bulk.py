"""Bulk analysis tab callbacks."""

import io
import os
import base64

import pandas as pd
from dash import Input, Output, State, no_update

from secactpy.app.utils import empty_figure


def _decode_csv(contents, filename):
    """Decode uploaded CSV/TSV to DataFrame."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    ext = filename.rsplit(".", 1)[-1].lower() if filename else "csv"
    sep = "," if ext == "csv" else "\t"
    return pd.read_csv(io.BytesIO(decoded), index_col=0, sep=sep)


def register_bulk_callbacks(app):
    """Register bulk analysis callbacks."""

    _bulk_cache = {}

    # --- Activity Change ---
    @app.callback(
        [Output("bulk-change-welcome", "style"),
         Output("bulk-change-results", "style"),
         Output("bulk-change-plot", "figure"),
         Output("bulk-change-table", "data"),
         Output("bulk-change-table", "columns"),
         Output("bulk-change-status", "children")],
        Input("bulk-change-btn", "n_clicks"),
        [State("bulk-treatment-upload", "contents"),
         State("bulk-treatment-upload", "filename"),
         State("bulk-control-upload", "contents"),
         State("bulk-control-upload", "filename")],
        prevent_initial_call=True,
    )
    def run_activity_change(n_clicks, treat_contents, treat_name, ctrl_contents, ctrl_name):
        if not treat_contents or not ctrl_contents:
            return (no_update,) * 5 + ("Upload both treatment and control files",)

        try:
            treatment = _decode_csv(treat_contents, treat_name)
            control = _decode_csv(ctrl_contents, ctrl_name)

            from secactpy import secact_activity_inference
            result = secact_activity_inference(
                treatment, control_profile=control,
                is_differential=False, verbose=False,
            )

            zscore_df = result.get("zscore", pd.DataFrame())
            if zscore_df.empty:
                return (no_update,) * 5 + ("No results returned",)

            _bulk_cache["change"] = zscore_df

            from secactpy.visualization import activity_change_bar
            if zscore_df.shape[1] == 1:
                zscore_series = zscore_df.iloc[:, 0]
            else:
                zscore_series = zscore_df.mean(axis=1)
            fig = activity_change_bar(zscore_series)

            display = zscore_df.reset_index()
            display.columns = ["Protein"] + list(display.columns[1:])

            return (
                {"display": "none"},
                {"display": "block"},
                fig,
                display.round(4).to_dict("records"),
                [{"name": c, "id": c} for c in display.columns],
                "Complete!",
            )
        except Exception as e:
            return (no_update,) * 5 + (f"Error: {e}",)

    # --- Cohort Survival ---
    @app.callback(
        [Output("bulk-survival-welcome", "style"),
         Output("bulk-survival-results", "style"),
         Output("bulk-lollipop-plot", "figure"),
         Output("bulk-survival-table", "data"),
         Output("bulk-survival-table", "columns"),
         Output("bulk-survival-protein", "options"),
         Output("bulk-survival-protein", "value"),
         Output("bulk-survival-status", "children")],
        Input("bulk-survival-btn", "n_clicks"),
        [State("bulk-expr-upload", "contents"),
         State("bulk-expr-upload", "filename"),
         State("bulk-clinical-upload", "contents"),
         State("bulk-clinical-upload", "filename")],
        prevent_initial_call=True,
    )
    def run_survival(n_clicks, expr_contents, expr_name, clin_contents, clin_name):
        if not expr_contents or not clin_contents:
            return (no_update,) * 7 + ("Upload both expression and clinical files",)

        try:
            expr_df = _decode_csv(expr_contents, expr_name)
            clinical = _decode_csv(clin_contents, clin_name)

            from secactpy import secact_activity_inference
            result = secact_activity_inference(expr_df, verbose=False)
            activity = result.get("zscore", pd.DataFrame())

            if activity.empty:
                return (no_update,) * 7 + ("No activity results",)

            _bulk_cache["activity"] = activity
            _bulk_cache["clinical"] = clinical

            # Cox regression (simplified — compute correlation with survival)
            from scipy.stats import spearmanr
            common = activity.columns.intersection(clinical.index)
            if len(common) < 5:
                return (no_update,) * 7 + ("Too few overlapping samples between expression and clinical",)

            time_col = clinical.columns[clinical.columns.str.lower().isin(["time"])][0] if any(clinical.columns.str.lower() == "time") else clinical.columns[0]
            risk_scores = {}
            for protein in activity.index:
                vals = activity.loc[protein, common].astype(float)
                surv = clinical.loc[common, time_col].astype(float)
                r, p = spearmanr(vals, surv)
                risk_scores[protein] = r
            risk_series = pd.Series(risk_scores).sort_values()
            _bulk_cache["risk"] = risk_series

            from secactpy.visualization import risk_lollipop
            fig = risk_lollipop(risk_series)

            proteins = risk_series.index.tolist()
            display = pd.DataFrame({"Protein": proteins, "Risk Score": risk_series.values}).round(4)

            return (
                {"display": "none"},
                {"display": "block"},
                fig,
                display.to_dict("records"),
                [{"name": c, "id": c} for c in display.columns],
                [{"label": p, "value": p} for p in proteins],
                proteins[0] if proteins else None,
                "Complete!",
            )
        except Exception as e:
            return (no_update,) * 7 + (f"Error: {e}",)

    # Survival curve for selected protein
    @app.callback(
        Output("bulk-survival-plot", "figure"),
        Input("bulk-survival-protein", "value"),
        prevent_initial_call=True,
    )
    def update_survival_curve(protein):
        if not protein or "activity" not in _bulk_cache or "clinical" not in _bulk_cache:
            return empty_figure("Run survival analysis first")

        try:
            import plotly.graph_objects as go
            activity = _bulk_cache["activity"]
            clinical = _bulk_cache["clinical"]
            common = activity.columns.intersection(clinical.index)

            time_col = clinical.columns[clinical.columns.str.lower().isin(["time"])][0] if any(clinical.columns.str.lower() == "time") else clinical.columns[0]
            event_col = clinical.columns[clinical.columns.str.lower().isin(["event"])][0] if any(clinical.columns.str.lower() == "event") else clinical.columns[1]

            vals = activity.loc[protein, common].astype(float)
            median_val = vals.median()
            high = vals[vals >= median_val].index
            low = vals[vals < median_val].index

            fig = go.Figure()
            for group, label, color in [(high, "High", "#e74c3c"), (low, "Low", "#3498db")]:
                times = clinical.loc[group, time_col].astype(float).sort_values()
                n = len(times)
                survival = [(n - i) / n for i in range(n)]
                fig.add_trace(go.Scatter(x=times.values, y=survival,
                                          mode="lines", name=f"{label} (n={n})",
                                          line=dict(color=color)))

            fig.update_layout(
                title=f"Survival: {protein}",
                xaxis_title="Time", yaxis_title="Survival Probability",
                template="plotly_white",
            )
            return fig
        except Exception as e:
            return empty_figure(f"Error: {e}")
