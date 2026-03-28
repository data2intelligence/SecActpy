"""Spatial tab callbacks."""

from dash import Input, Output, State, no_update

from secactpy.app.utils import empty_figure


def register_spatial_callbacks(app):
    """Register all spatial tab callbacks with the Dash app."""

    @app.callback(
        [Output("spatial-welcome", "style"),
         Output("spatial-plot-area", "style"),
         Output("spatial-controls", "style"),
         Output("spatial-feature", "options"),
         Output("spatial-feature", "value"),
         Output("spatial-data-store", "data"),
         Output("spatial-inference-btn", "style")],
        [Input("spatial-demo-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def load_demo(n_clicks):
        """Load bundled demo dataset."""
        try:
            # Phase 2: wire to actual demo data via spatial-gpu
            features = ["Feature loading requires spatial-gpu demo data"]
            return (
                {"display": "none"},
                {"display": "block"},
                {"display": "block"},
                [{"label": f, "value": f} for f in features],
                features[0] if features else None,
                {"loaded": True},
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
        """Render spatial plot using spatial-gpu."""
        if not data_store or not feature:
            return empty_figure("No data loaded")
        try:
            # Phase 2: call spatialgpu.visualization.spatial_scatter()
            return empty_figure(f"Spatial plot for {feature} (spatial-gpu integration pending)")
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
        """Render statistics plot using secactpy.visualization."""
        if not data_store or not feature:
            return empty_figure("No data loaded")
        try:
            # Phase 2: call secactpy.visualization functions
            return empty_figure(f"Statistics for {feature} (pending data integration)")
        except Exception as e:
            return empty_figure(f"Error: {e}")
