import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

def layout():
    """
    Returns the layout for the image viewer component.
    """
    return html.Div(
        [
            # Title with light blue background
            html.Div(
                children=[  
                    html.H1(
                        "1D Spectra Visualization",
                        style={
                            'textAlign': 'center',
                            'color': 'white',
                            'padding': '5px',
                            'margin': '0',
                        }
                    ),
                ],
                style={
                    'backgroundColor': '#007bff',
                    'borderRadius': '10px',
                    'marginBottom': '20px',  # Space below the title
                    "border": "2px solid black"
                }
            ),
            dbc.Card(
                id="image-card",
                children=[
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    # Scatter plot with its own border
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="scatter",
                                                figure=go.Figure(
                                                    go.Scattergl(mode="markers"),
                                                    layout=go.Layout(
                                                        autosize=True,
                                                        margin=go.layout.Margin(
                                                            l=20,
                                                            r=20,
                                                            b=20,
                                                            t=20,
                                                            pad=0,
                                                        ),
                                                    ),
                                                ),
                                            ),
                                            style={
                                                "border": "2px solid black",
                                                "padding": "10px",
                                                "border-radius": "10px"
                                            }  # Styling for scatter plot border
                                        ),
                                        width=6,  # Size of scatter plot
                                    ),
                                    # Heatmap with its own border
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="heatmap",
                                                figure=go.Figure(
                                                    go.Heatmap(),
                                                    layout=go.Layout(
                                                        autosize=True,
                                                        margin=go.layout.Margin(
                                                            l=20,
                                                            r=20,
                                                            b=20,
                                                            t=20,
                                                            pad=0,
                                                        ),
                                                    ),
                                                ),
                                            ),
                                            style={
                                                "border": "2px solid black",
                                                "padding": "10px",
                                                "border-radius": "10px"
                                            }  # Styling for heatmap border
                                        ),
                                        width=6,  # Size of heatmap
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                style={"padding": "1em", "border-radius": "10px"}  # Padding for the card
            )
        ]
    )
