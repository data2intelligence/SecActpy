"""Shared Dash helpers."""

import plotly.graph_objects as go


def empty_figure(message: str) -> go.Figure:
    """Placeholder figure for missing data or errors."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
