"""SecActpy Dash application entry point."""

import dash
import dash_bootstrap_components as dbc
from dash import html

from secactpy.app.config import UI_COLORS
from secactpy.app.layouts.spatial import spatial_layout
from secactpy.app.callbacks.spatial import register_spatial_callbacks
from secactpy.app.layouts.inference import inference_layout
from secactpy.app.callbacks.inference import register_inference_callbacks


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div([
        html.Div([
            html.H2("SecAct", style={"display": "inline", "marginTop": "0"}),
            html.Span("Secreted Protein Activity Analysis",
                       style={"marginLeft": "15px", "opacity": "0.8"}),
        ], style={
            "backgroundColor": UI_COLORS["primary"],
            "color": "white", "padding": "15px", "marginBottom": "20px",
        }),

        dbc.Tabs([
            dbc.Tab(spatial_layout(), label="Spatial"),
            dbc.Tab(inference_layout(), label="Run Inference"),
        ]),
    ])

    register_spatial_callbacks(app)
    register_inference_callbacks(app)
    return app
