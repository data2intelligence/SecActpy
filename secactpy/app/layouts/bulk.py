"""Bulk analysis tab layout — activity change and cohort survival."""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from secactpy.app.config import UI_COLORS


def bulk_layout():
    return dbc.Container([
        dbc.Tabs([
            # Sub-tab 1: Activity Change
            dbc.Tab(label="Activity Change", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Activity Change",
                                           style={"color": UI_COLORS["primary"]}),
                            dbc.CardBody([
                                html.P("Compare activity between treatment and control.",
                                       className="text-muted"),
                                dcc.Upload(id="bulk-treatment-upload",
                                           children=html.Div(["Treatment expression"]),
                                           style={"borderWidth": "1px", "borderStyle": "dashed",
                                                   "borderRadius": "5px", "textAlign": "center",
                                                   "padding": "15px", "cursor": "pointer"}),
                                html.Br(),
                                dcc.Upload(id="bulk-control-upload",
                                           children=html.Div(["Control expression"]),
                                           style={"borderWidth": "1px", "borderStyle": "dashed",
                                                   "borderRadius": "5px", "textAlign": "center",
                                                   "padding": "15px", "cursor": "pointer"}),
                                html.Small("CSV/TSV, log2(x+1) transformed", className="text-muted"),
                                html.Hr(),
                                dbc.Button("Run Activity Change", id="bulk-change-btn",
                                           color="primary", className="w-100"),
                                html.Div(id="bulk-change-status", className="mt-2 text-muted"),
                            ]),
                        ]),
                    ], width=4),
                    dbc.Col([
                        html.Div(id="bulk-change-welcome", children=[
                            html.Div([
                                html.H3("Activity Change Analysis"),
                                html.P("Upload treatment and control expression data."),
                            ], style={"textAlign": "center", "padding": "80px 0"}),
                        ]),
                        html.Div(id="bulk-change-results", style={"display": "none"}, children=[
                            dcc.Loading(dcc.Graph(id="bulk-change-plot", style={"height": "400px"})),
                            html.Hr(),
                            dash_table.DataTable(id="bulk-change-table", page_size=15,
                                                 style_table={"overflowX": "auto"},
                                                 export_format="csv"),
                        ]),
                    ], width=8),
                ], className="mt-3"),
            ]),

            # Sub-tab 2: Cohort Survival
            dbc.Tab(label="Cohort Survival", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Cohort Survival",
                                           style={"color": UI_COLORS["primary"]}),
                            dbc.CardBody([
                                html.P("Link activity to clinical outcomes.", className="text-muted"),
                                dcc.Upload(id="bulk-expr-upload",
                                           children=html.Div(["Expression matrix"]),
                                           style={"borderWidth": "1px", "borderStyle": "dashed",
                                                   "borderRadius": "5px", "textAlign": "center",
                                                   "padding": "15px", "cursor": "pointer"}),
                                html.Br(),
                                dcc.Upload(id="bulk-clinical-upload",
                                           children=html.Div(["Clinical data (Time, Event)"]),
                                           style={"borderWidth": "1px", "borderStyle": "dashed",
                                                   "borderRadius": "5px", "textAlign": "center",
                                                   "padding": "15px", "cursor": "pointer"}),
                                html.Small("Clinical: must have 'Time' and 'Event' columns",
                                           className="text-muted"),
                                html.Hr(),
                                dbc.Button("Run Survival Analysis", id="bulk-survival-btn",
                                           color="primary", className="w-100"),
                                html.Div(id="bulk-survival-status", className="mt-2 text-muted"),
                                html.Hr(),
                                dbc.Label("Protein for Survival Curve"),
                                dcc.Dropdown(id="bulk-survival-protein"),
                            ]),
                        ]),
                    ], width=4),
                    dbc.Col([
                        html.Div(id="bulk-survival-welcome", children=[
                            html.Div([
                                html.H3("Cohort Survival Analysis"),
                                html.P("Upload expression and clinical data."),
                            ], style={"textAlign": "center", "padding": "80px 0"}),
                        ]),
                        html.Div(id="bulk-survival-results", style={"display": "none"}, children=[
                            dcc.Loading(dcc.Graph(id="bulk-lollipop-plot", style={"height": "400px"})),
                            html.Hr(),
                            dcc.Loading(dcc.Graph(id="bulk-survival-plot", style={"height": "400px"})),
                            html.Hr(),
                            dash_table.DataTable(id="bulk-survival-table", page_size=15,
                                                 style_table={"overflowX": "auto"},
                                                 export_format="csv"),
                        ]),
                    ], width=8),
                ], className="mt-3"),
            ]),
        ]),
    ], fluid=True)
