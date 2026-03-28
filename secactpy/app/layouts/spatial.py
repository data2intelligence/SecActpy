"""Spatial tab layout."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from secactpy.app.config import UI_COLORS


def spatial_layout():
    return dbc.Container([
        dbc.Row([
            # Sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Dataset", style={"color": UI_COLORS["primary"]}),
                    dbc.CardBody([
                        dcc.Upload(
                            id="spatial-upload",
                            children=html.Div(["Drag & drop or ", html.A("select file")]),
                            style={
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "5px", "textAlign": "center",
                                "padding": "20px", "cursor": "pointer",
                            },
                        ),
                        html.Small("Accepts .h5ad, .rds, .csv", className="text-muted"),
                        html.Hr(),
                        dbc.Button("Load Demo (Visium)", id="spatial-demo-btn",
                                   color="info", className="w-100 mb-2"),
                        dbc.Button("Run SecAct Inference", id="spatial-inference-btn",
                                   color="warning", className="w-100",
                                   style={"display": "none"}),
                    ]),
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Visualization", style={"color": UI_COLORS["primary"]}),
                    dbc.CardBody([
                        dbc.Label("Display Type"),
                        dcc.Dropdown(
                            id="spatial-type",
                            options=[
                                {"label": "SecAct Activity", "value": "activity"},
                                {"label": "Gene Expression", "value": "expression"},
                                {"label": "Cell Fraction", "value": "fraction"},
                            ],
                            value="activity",
                        ),
                        dbc.Label("Feature", className="mt-2"),
                        dcc.Dropdown(id="spatial-feature"),
                        dbc.Label("Point Size", className="mt-2"),
                        dcc.Slider(id="spatial-pointsize", min=1, max=20, value=5, step=1),
                    ]),
                ], id="spatial-controls", style={"display": "none"}),
            ], width=3),
            # Main panel
            dbc.Col([
                html.Div(id="spatial-welcome", children=[
                    html.Div([
                        html.H2("Spatial Visualization"),
                        html.P("Upload data or load the demo to explore spatial SecAct activity."),
                        html.P("Supports: Visium, VisiumHD, CosMx, Xenium",
                               className="text-muted"),
                    ], style={"textAlign": "center", "padding": "100px 0"}),
                ]),
                html.Div(id="spatial-plot-area", style={"display": "none"}, children=[
                    dcc.Loading(dcc.Graph(id="spatial-plot", style={"height": "600px"})),
                    html.Hr(),
                    dcc.Loading(dcc.Graph(id="spatial-stats-plot", style={"height": "400px"})),
                ]),
            ], width=9),
        ]),
        dcc.Store(id="spatial-data-store"),
    ], fluid=True)
