#Import necessary modules
from dash import Input, Output, State, callback
from utils.plot_utils import generate_scatter_plot, working_generate_line_plot
from utils.data_utils import BASE_DATA, BASE
import pandas as pd
import dash
import plotly.graph_objects as go
import os
import base64


base = BASE


@callback(
    Output("scatter", "figure"),
    Input("latent-vectors", "data"),
    Input("latent-vectors2", "data"),
    State("scatter", "figure"),
)
def update_scatter_plot(data_store: list, data_store2: list, current_figure: dict):
    """
    Updates the scatter plot with the latent vectors.
    :param data_store: List of latent vectors.
    :param data_store2: List of benchmark latent vectors.
    :param current_figure: Current scatter plot figure.
    :return: Updated scatter plot figure.
    """
    # Assume data_store contains the latent vectors
    latent_vectors = data_store  
    bm_latent_vectors = data_store2
    
    if latent_vectors:
        # Generate the new scatter plot figure
        fig = generate_scatter_plot(latent_vectors,bm_latent_vectors, 2)

        # Automatically adjust axes to fit data
        fig.update_xaxes(autorange=True) 
        fig.update_yaxes(autorange=True) 
        
        # Add title and styling to the scatter plot
        fig.update_layout(
            title={
                'text': "SCATTER PLOT OF LATENT VECTORS",
                'font': {'size': 20, 'color': 'darkblue'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(t=80),
            paper_bgcolor='rgba(173, 216, 230, 0.5)',
            plot_bgcolor='rgba(0,0,0,0)'
        )  
        return fig
    else:
        # Return the current figure if no data
        return current_figure  
@callback(
    Output("heatmap", "figure"),
    Input("scatter", "selectedData"),
    Input("scatter", "clickData"),
    Input("mean-std-toggle", "value"),
    State("data-store", "data"),
    State("data-store2", "data"),
    prevent_initial_call=True,
)
def update_heatmap(
    selected_data: dict,
    click_data: dict,
    display_option: str,
    data_store: list,
    data_store2: list
):
    """
    Updates the heatmap based on user interaction with the scatter plot.
    :param selected_data: Data from points selected in scatter plot.
    :param click_data: Data from points clicked in scatter plot.
    :param display_option: Option to display mean or standard deviation.
    :param data_store: Primary data store.
    :param data_store2: Secondary data store.
    :return: Updated heatmap figure.
    """

# Check if data_store is not empty
    if not data_store:
        print("Data store is empty.")
        return {}  # Return an empty figure

    # Extract intensity values based on selected points
    selected_intensities = []
    file_names = []
    
    # Helper functions for data extraction
    def extract_data_sigscan(point_index, customdata):
        if point_index < len(data_store):  # Ensure the index is valid
            selected_intensities.append(data_store[point_index]) 
        else:
            print(f"Invalid index: {point_index}")

    def extract_data_non_sigscan(point_index, customdata):
        if point_index < len(data_store2):  # Ensure the index is valid
            selected_intensities.append(data_store[point_index]) 
        else:
            print(f"Invalid index: {point_index}")
            
    def extract_bm_data_sigscan(point_index, customdata):
        if point_index < len(data_store):  # Ensure the index is valid
            selected_intensities.append(data_store2[point_index]) 
        else:
            print(f"Invalid index: {point_index}")

    def extract_bm_data_non_sigscan(point_index, customdata):
        if point_index < len(data_store2):  # Ensure the index is valid
            selected_intensities.append(data_store2[point_index]) 
        else:
            print(f"Invalid index: {point_index}")
    
    if selected_data and "points" in selected_data:
        
        for point in selected_data["points"]:
            point_index = point['pointIndex']
            
            customdata = point['customdata'][0]  # Get the first element of customdata
            
            if 'SigScan' in customdata:
                file_name = customdata.split('File: ')[1].split('<')[0]
                last_digit = file_name[-1]
                try:
                    if int(last_digit) >=0:
                        
                        file_names.append(file_name)
                        extract_data_sigscan(point_index,customdata)
                        
                except:
                    file_names.append(file_name)
                    extract_bm_data_sigscan(point_index,customdata)
                
            elif '.txt' in customdata:
                file_name = customdata.split('File: ')[1].split('<')[0]
                last_digit = file_name[-1]
                try:
                    if int(last_digit) >=0:
                        file_names.append(file_name)
                        extract_data_non_sigscan(point_index,customdata)
                    
                except:
                    file_names.append(file_name)
                    extract_bm_data_non_sigscan(point_index,customdata)

    elif click_data and "points" in click_data:
        # Handle click event if no selected points
        selected_file_name = click_data["points"][0]["customdata"][0]
        # You can extract intensities based on the clycked file name if needed

    
    # Return empty if no intensities selected
    if not selected_intensities:
        print("No intensities selected.")
        return {}
    

    # Generate and return the updated heatmap figure
    fig = working_generate_line_plot(selected_intensities, file_names, display_option)
    

    fig.update_layout(
        title={
            'text': f"{display_option.upper()} PLOT OF RAW DATA",
            'font': {'size': 20, 'color': 'darkblue'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=80),
        paper_bgcolor='rgba(173, 216, 230, 0.5)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig



@callback(
    [Output('scatter', 'figure', allow_duplicate=True), Output('heatmap', 'figure', allow_duplicate=True), Output('download-data-button', 'href')],
    [Input('reset-button', 'n_clicks'), 
     Input('generate-data-button', 'n_clicks'), 
     Input('download-data-button', 'n_clicks'), 
     Input('browse-data-upload', 'contents')],  # Use 'contents' here instead of 'n_clicks'
    [State('latent-vectors', 'data'),
     State('latent-vectors2', 'data'), 
     State('scatter', 'figure'), 
     State('data-store', 'data'),
     State('data-store2', 'data')],
    prevent_initial_call=True
)
def update_plots(reset_n_clicks, load_n_clicks, download_n_clicks, uploaded_file_contents, latent_vectors, latent_vectors2, current_scatter, data, bm_data):
    """
    Updates both the scatter and heatmap plots, handles file upload and reset functionality.
    """
    ctx = dash.callback_context
    path1 = os.path.join(base, 'data', 'latent_vectors_relevant_data1.parquet')
    path2 = os.path.join(base, 'data', 'relevant_data1.parquet')
    import pandas as pd
    df = pd.read_parquet(path1)
    df1 = pd.read_parquet(path2)

    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_input == 'reset-button':
        # Reset scatter plot
        scatter_fig = go.Figure(
            go.Scattergl(mode="markers"),
            layout=go.Layout(
                autosize=True,
                margin=go.layout.Margin(
                    l=20, r=20, b=20, t=80, pad=0,  # Increased top margin
                ),
                title={
                    'text': "SCATTER PLOT OF LATENT VECTORS",  # Reset title for scatter
                    'font': {'size': 20, 'color': 'darkblue'},  # Increased font size
                    'y': 0.95,  # Position title at the top
                    'x': 0.5,   # Center title horizontally
                    'xanchor': 'center',  # Centering the title
                    'yanchor': 'top'  # Anchoring the title at the top
                }
            )
        )

        # Reset heatmap plot
        heatmap_fig = go.Figure(
            go.Heatmap(),
            layout=go.Layout(
                autosize=True,
                margin=go.layout.Margin(
                    l=20, r=20, b=20, t=80, pad=0,  # Increased top margin
                ),
                title={
                    'text': "MEAN PLOT OF RAW DATA",  # Reset title for heatmap
                    'font': {'size': 20, 'color': 'darkblue'},  # Increased font size
                    'y': 0.95,  # Position title at the top
                    'x': 0.5,   # Center title horizontally
                    'xanchor': 'center',  # Centering the title
                    'yanchor': 'top'  # Anchoring the title at the top
                }
            )
        )

        # Reset data
        df = df.iloc[0:1]
        df1 = df1.iloc[0:1]

        df.to_parquet(path1)
        df1.to_parquet(path2)
        return scatter_fig, heatmap_fig, dash.no_update

    elif uploaded_file_contents is not None:
        # Process uploaded file contents (base64 encoded)
        content_type, content_string = uploaded_file_contents.split(',')
        decoded_file = base64.b64decode(content_string)

        # Example of how you might handle file processing (e.g., CSV or Excel)
        # For example, let's assume it's a CSV file:
        try:
            from io import StringIO
            import pandas as pd
            # Assuming the uploaded file is CSV format
            decoded_file_str = StringIO(decoded_file.decode('utf-8'))
            uploaded_data_df = pd.read_csv(decoded_file_str)

            # You can now use `uploaded_data_df` in your callback
            # For example, update scatter and heatmap with the new data
            scatter_fig = update_scatter_plot(latent_vectors, latent_vectors2, current_scatter)
            heatmap_fig = update_heatmap(None, None, 'mean', data, bm_data)

            return scatter_fig, heatmap_fig, dash.no_update
        except Exception as e:
            print(f"Error processing file: {e}")
            return dash.no_update, dash.no_update, dash.no_update

    elif triggered_input == 'generate-data-button':
        # Update scatter plot with data
        scatter_fig = update_scatter_plot(latent_vectors, latent_vectors2, current_scatter)

        # Update heatmap with data
        heatmap_fig = update_heatmap(None, None, 'mean', data, bm_data)

        return scatter_fig, heatmap_fig, dash.no_update

    elif triggered_input == 'download-data-button':
        # Define the paths of the files to be downloaded
        file_path_1 = '/Users/monikachoudhary/downloads/latent_vectors_relevant_data1.parquet'
        file_path_2 = '/Users/monikachoudhary/downloads/relevant_data1.parquet'

        df.to_parquet(file_path_1)
        df1.to_parquet(file_path_2)
        # Return the download link for the relevant data file
        return dash.no_update, dash.no_update, f'{file_path_1}?download=true'

    return dash.no_update, dash.no_update, dash.no_update
