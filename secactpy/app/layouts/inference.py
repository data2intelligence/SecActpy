"""Inference tab layout — upload data, run SecAct, view results."""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from secactpy.app.config import UI_COLORS


def inference_layout():
    return dbc.Container([
        dbc.Row([
            # Sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Upload Data", style={"color": UI_COLORS["primary"]}),
                    dbc.CardBody([
                        dcc.Upload(
                            id="inference-upload",
                            children=html.Div(["Drag & drop or ", html.A("select file")]),
                            style={
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "5px", "textAlign": "center",
                                "padding": "20px", "cursor": "pointer",
                            },
                        ),
                        html.Small("CSV, TSV, or H5AD (genes x samples)", className="text-muted"),
                        html.Hr(),
                        dbc.Label("Input type"),
                        dbc.RadioItems(
                            id="inference-input-type",
                            options=[
                                {"label": "Differential expression (logFC)", "value": "logFC"},
                                {"label": "Raw expression matrix", "value": "expression"},
                            ],
                            value="logFC",
                        ),
                        html.Hr(),
                        dbc.Button("Run SecAct Inference", id="inference-run-btn",
                                   color="primary", className="w-100",
                                   disabled=True),
                        html.Div(id="inference-status", className="mt-2 text-muted"),
                    ]),
                ]),
            ], width=4),
            # Results
            dbc.Col([
                html.Div(id="inference-welcome", children=[
                    html.Div([
                        html.H3("Run SecAct Inference"),
                        html.P("Upload expression data to infer secreted protein signaling activity."),
                        html.P("Rows = genes, Columns = samples/spots.",
                               className="text-muted"),
                    ], style={"textAlign": "center", "padding": "80px 0"}),
                ]),
                html.Div(id="inference-results-area", style={"display": "none"}, children=[
                    html.H4("Results", style={"color": UI_COLORS["primary"]}),
                    dcc.Loading(dash_table.DataTable(
                        id="inference-results-table",
                        page_size=15,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "5px"},
                        style_header={"fontWeight": "bold"},
                        export_format="csv",
                    )),
                    html.Br(),
                    dbc.Button("Download Results (CSV)", id="inference-download-btn",
                               color="success", className="me-2"),
                    dcc.Download(id="inference-download"),
                ]),
            ], width=8),
        ]),
        dcc.Store(id="inference-data-store"),
    ], fluid=True)
