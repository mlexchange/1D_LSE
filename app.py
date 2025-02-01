import dash_bootstrap_components as dbc
from dash import Dash, html, dcc  # Correct import
from dash.dependencies import Input, Output, State

from callbacks.image_viewer import update_scatter_plot, update_heatmap # noqa: F401
from callbacks.update_data import update_data  # noqa: F401
from components.controls import layout as controls_layout
from components.image_viewer import layout as image_panel
import dash
import plotly.graph_objects as go



app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
# Main layout
app.layout = html.Div(
    dbc.Container(
        children=[
            dcc.Store(id='data-store'),
            dcc.Store(id='latent-vectors'),
            dcc.Store(id='data-store2'),
            dcc.Store(id='latent-vectors2'),
            dcc.Store(id='energy-store'),
            dcc.Store(id='intensity-store'),
            dcc.Store(id='append-status'),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(),
                        width=2,
                        style={"display": "flex", "margin-top": "1em", "border-right": "2px solid #d3d3d3", "padding-right": "1em"},
                    ),
                    dbc.Col(
                        image_panel(), 
                        width=10,
                        style={"padding-left": "1em"},
                    ),
                ]
            ),
        ],
        fluid=True,  # Full width container
        style={"border": "2px solid black", "padding": "1em", "border-radius": "10px"}
    ),
    style={"height": "100vh", "width": "100vw"}  # Full viewport height and width
)



if __name__ == "__main__":
    app.run_server(port=8064, debug=True)