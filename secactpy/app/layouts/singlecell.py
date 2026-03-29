"""Single-cell tab layout — upload scRNA-seq data, run inference, visualize."""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from secactpy.app.config import UI_COLORS


def singlecell_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Upload scRNA-seq Data",
                                   style={"color": UI_COLORS["primary"]}),
                    dbc.CardBody([
                        dcc.Upload(
                            id="sc-upload",
                            children=html.Div(["Drag & drop or ", html.A("select .h5ad")]),
                            style={
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "5px", "textAlign": "center",
                                "padding": "15px", "cursor": "pointer",
                            },
                        ),
                        html.Small("AnnData .h5ad with cell type annotations",
                                   className="text-muted"),
                        html.Hr(),
                        dbc.Label("Cell type column"),
                        dbc.Input(id="sc-celltype-col", value="cell_type", type="text"),
                        html.Hr(),
                        dbc.Button("Run SC Inference", id="sc-run-btn",
                                   color="primary", className="w-100", disabled=True),
                        html.Div(id="sc-status", className="mt-2 text-muted"),
                    ]),
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Visualization", style={"color": UI_COLORS["primary"]}),
                    dbc.CardBody([
                        dbc.Label("Protein"),
                        dcc.Dropdown(id="sc-protein"),
                        dbc.Label("Plot Type", className="mt-2"),
                        dcc.Dropdown(
                            id="sc-plot-type",
                            options=[
                                {"label": "Activity Bar", "value": "bar"},
                                {"label": "Activity Heatmap", "value": "heatmap"},
                                {"label": "Cell Type Distribution", "value": "celltype"},
                            ],
                            value="bar",
                        ),
                    ]),
                ], id="sc-controls", style={"display": "none"}),
            ], width=3),
            dbc.Col([
                html.Div(id="sc-welcome", children=[
                    html.Div([
                        html.H3("Single-Cell SecAct Analysis"),
                        html.P("Upload scRNA-seq data and run cell-type level inference."),
                        html.P("Provide an AnnData (.h5ad) with cell type annotations.",
                               className="text-muted"),
                    ], style={"textAlign": "center", "padding": "80px 0"}),
                ]),
                html.Div(id="sc-results-area", style={"display": "none"}, children=[
                    dcc.Loading(dcc.Graph(id="sc-plot", style={"height": "500px"})),
                    html.Hr(),
                    dcc.Loading(dash_table.DataTable(
                        id="sc-results-table",
                        page_size=15,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "5px"},
                        style_header={"fontWeight": "bold"},
                        export_format="csv",
                    )),
                ]),
            ], width=9),
        ]),
        dcc.Store(id="sc-data-store"),
    ], fluid=True)
