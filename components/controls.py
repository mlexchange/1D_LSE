import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

def layout():
    return html.Div(
        [
            html.Div(
                html.P("CONTROLS", style={
                    "font-weight": "bold",
                    "font-size": "1.2em",
                    "margin-bottom": "0",
                    "color": "white"
                }),
                style={
                    "background-color": "#007bff",
                    "padding": "2px",
                    "border": "2px solid black",
                    "border-radius": "10px", 
                    "text-align": "center", 
                    "height": "35px",
                    "line-height": "35px",
                }
            ),
            dcc.RadioItems(
                id="mean-std-toggle",
                options=[
                    {"label": "Original Plot", "value": "original"},
                    {"label": "Waterfall Plot", "value": "waterfall"},
                    {"label": "Median with Inter-Quartile", "value": "median iqr"},
                    {"label": "Mean with Min-Max", "value": "mean minmax"},
                    # {"label": "Peaks of Spectras", "value": "peaks"},
                    {"label": "Peak Fitted Plot", "value": "gaussian"}
                ],
                value="original plots",
                style={
                    "min-width": "250px",
                    "padding-bottom": "1em",
                    "font-size": "1.1em"
                },
                className="mb-2",
            ),
            # Load Data button
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Load data", 
                            id="generate-data-button", 
                            color="primary", 
                            style={"border": "2px solid black", "width": "100%"}
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            
                        # Browse Data functionality
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Upload(
                            id="browse-data-upload",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Browse")]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "margin": "10px 0",
                                "background-color": "#f9f9f9"
                            },
                            multiple=False  # Allow single file upload only
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id='spectrum-dropdown',
                            placeholder="Select a benchmark filename",
                            # options=[{'label': f'Spectrum {i}', 'value': i} for i in range(10)],
                            value=0,
                            style={'width': '100%'},
                            clearable=False,
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            # Amplitude Shift
            dbc.Row(
                [
                    dbc.Col(html.Label("Amplitude Shift"), width=12),
                    dbc.Col(
                        dcc.Slider(
                            id='amplitude-shift',
                            min=0.0,
                            max=5.0,
                            step=0.5,
                            value=0.0,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode='drag',
                            marks={i/2 : '' for i in range(0, 11)},
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            # Mu Shift
            dbc.Row(
                [
                    dbc.Col(html.Label("Mu Shift"), width=12),
                    dbc.Col(
                        dcc.Slider(
                            id='mean-shift',
                            min=-10.0,
                            max=10.0,
                            step=2.0,
                            value=0.0,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode='drag',
                            marks={i : '' for i in range(-10, 11, 2)},
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            # Sigma Shift
            dbc.Row(
                [
                    dbc.Col(html.Label("Sigma Shift"), width=12),
                    dbc.Col(
                        dcc.Slider(
                            id='sigma-shift',
                            min=-0.005,
                            max=0.005,
                            step=0.001,
                            value=0.0,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode='drag',
                            marks={i/1000: '' for i in range(-5, 6)},
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            # Noise Shift
            dbc.Row(
                [
                    dbc.Col(html.Label("Noise Shift"), width=12),
                    dbc.Col(
                        dcc.Slider(
                            id='noise-shift',
                            min=0.00,
                            max=1.00,
                            step=0.1,
                            value=0.00,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode='drag',
                            marks={i / 10: '' for i in range(0, 11)},
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Append Data", 
                            id="append-data-button", 
                            color="primary", 
                            style={"border": "2px solid black", "width": "100%"}
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Download Data", 
                            id="download-data-button", 
                            color="primary", 
                            style={"border": "2px solid black", "width": "100%"}
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Reset", 
                            id="reset-button", 
                            color="primary", 
                            style={"border": "2px solid black", "width": "100%"}
                        ),
                        width=12
                    ),
                ],
                className="mb-3"
            ),
            dcc.Store("data", data=[]),
        ],
        style={
            "border": "2px solid black", 
            "padding": "1em", 
            "border-radius": "10px"  
        } 
    )
