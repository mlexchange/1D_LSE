import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import io,base64
from scipy.signal import find_peaks
plt.switch_backend('Agg')

# Set Matplotlib backend to 'Agg' for non-interactive plotting
def working_generate_heatmap_plot(selected_spectra, control=1):
    if len(selected_spectra) == 0:
        return go.Figure()  # Return an empty figure if no data is selected

    # Determine the maximum length of the spectra
    max_length = max(len(spectrum) for spectrum in selected_spectra)

    # Pad the sequences to ensure uniform length
    padded_spectra = np.array([
        np.pad(spectrum, (0, max_length - len(spectrum)), mode='constant', constant_values=0)
        for spectrum in selected_spectra
    ])

    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=padded_spectra.T,  # Transpose for correct orientation
        colorscale='Viridis',  # You can change this to any colorscale you prefer
        colorbar=dict(title='Intensity'),
    ))

    fig.update_layout(
        title="Intensity Heatmap",
        xaxis=dict(title="Spectrum Index"),
        yaxis=dict(title="Selected Spectra"),
        autosize=True,
        margin=dict(l=20, r=20, b=20, t=20, pad=0),
    )

    return fig

def working_generate_line_plot(selected_spectra, file_names, display_option):

    if display_option == "original":
        fig = go.Figure()

        # Plot each spectrum's intensity
        for i, spectrum in enumerate(selected_spectra):
            fig.add_trace(go.Scatter(
                y=spectrum,  # Intensity values
                mode='lines',  # Connect points with lines
                name=file_names[i],  # Name for the legend
                line=dict(width=2)  # Line width for visibility
            ))

        # Update layout for better visualization
        fig.update_layout(
            # title="Intensity vs. Index",
            xaxis_title="Index",
            yaxis_title="Intensity",
            legend_title="Spectra",
            template='plotly',  # You can change this to other templates if desired
            margin=dict(l=20, r=20, b=20, t=20, pad=0),
            autosize=True
        )

        return fig
    
    elif display_option == "waterfall":
        print("Block1")
        fig = go.Figure()

        # Initialize cumulative height for stacking intensities vertically
        cumulative_height = 0

        # Loop through each spectrum's intensity
        for i, spectrum in enumerate(selected_spectra):
            intensity = np.array(spectrum)  # Ensure the spectrum is a numpy array
            x_positions = np.arange(len(intensity))  # X positions based on index

            # Add trace for the current spectrum, stacking with cumulative offset
            fig.add_trace(go.Scatter(
                y=intensity + cumulative_height,  # Adjust intensity by cumulative height
                x=x_positions,  # X values are the index of the spectrum points
                mode='lines',  # Connect points with lines
                name=f'Spec.{i + 1}',  # Name for the legend
                line=dict(width=1.5)  # Line width for visibility
            ))

            # Update cumulative height for the next spectrum to avoid overlapping
            cumulative_height += np.max(intensity) + 0.5  # Adding some space between plots

        # Customize x-axis ticks to match the example
        tick_positions = np.linspace(0, len(selected_spectra[0]), 4)  # Select 4 equally spaced ticks
        tick_labels = [f'{int(pos):.0f}' for pos in tick_positions]  # Format labels as integers

        # Update layout for better visualization
        fig.update_layout(
            title="Vertically Stacked Waterfall Plot of Spectra",
            xaxis_title="Index",
            yaxis_title="Intensity (Stacked)",
            xaxis=dict(tickvals=tick_positions, ticktext=tick_labels),  # Set custom x-axis ticks
            template='plotly_white',  # Light background template
            margin=dict(l=40, r=40, b=40, t=40, pad=0),
            autosize=True
        )

        return fig
    
    elif display_option == 'median iqr':


        # Find the maximum length among the spectra
        max_length = max(len(spectrum) for spectrum in selected_spectra)

        # Pad all spectra to the same length
        padded_spectra = [np.pad(spectrum, (0, max_length - len(spectrum)), 'constant', constant_values=np.nan) for spectrum in selected_spectra]

        # Convert the padded spectra into a NumPy array
        spectra_array = np.array(padded_spectra)

        # Calculate median and interquartile range (IQR), ignoring NaN values
        median_spectrum = np.nanmedian(spectra_array, axis=0)
        q1_spectrum = np.nanpercentile(spectra_array, 25, axis=0)
        q3_spectrum = np.nanpercentile(spectra_array, 75, axis=0)

        # Create the plot
        fig = go.Figure()

        # Plot IQR as a filled area
        fig.add_trace(go.Scatter(
            x=np.arange(max_length),  # X-axis (assuming index)
            y=q3_spectrum,  # Upper bound of the IQR
            mode='lines',
            line=dict(color='lightgray'),  # Light gray for IQR
            fill=None,
            showlegend=False  # No legend for this trace
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(max_length),  # X-axis (assuming index)
            y=q1_spectrum,  # Lower bound of the IQR
            mode='lines',
            line=dict(color='lightgray'),  # Light gray for IQR
            fill='tonexty',  # Fill the region between q1 and q3
            fillcolor='lightgray',
            showlegend=False  # No legend for this trace
        ))

        # Plot the median intensity as a separate line
        fig.add_trace(go.Scatter(
            x=np.arange(max_length),  # X-axis (assuming index)
            y=median_spectrum,  # Median intensity
            mode='lines',
            name='Median Intensity',
            line=dict(width=3, color='blue')  # Customize line appearance
        ))

        # Plot each individual spectrum's intensity
        for i, spectrum in enumerate(padded_spectra):
            fig.add_trace(go.Scatter(
                y=spectrum,  # Intensity values
                mode='lines',  # Connect points with lines
                name=f'Spec.{i + 1}',  # Name for the legend
                line=dict(width=1, dash='dot'),  # Dotted line for individual spectra
                opacity=0.3  # Lower opacity for individual spectra
            ))

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Intensity",
            legend_title="Spectra",
            template='plotly',
            margin=dict(l=20, r=20, b=20, t=20, pad=0),
            autosize=True
        )

        return fig
    
    
    elif display_option == 'mean minmax':
        
        max_length = max(len(spectrum) for spectrum in selected_spectra)

        # Pad all spectra to the same length
        padded_spectra = [np.pad(spectrum, (0, max_length - len(spectrum)), 'constant', constant_values=np.nan) for spectrum in selected_spectra]

        # Convert the padded spectra into a NumPy array
        spectra_array = np.array(padded_spectra)

        # Calculate mean and min-max range, ignoring NaN values
        mean_spectrum = np.nanmean(spectra_array, axis=0)
        min_spectrum = np.nanmin(spectra_array, axis=0)
        max_spectrum = np.nanmax(spectra_array, axis=0)

        # Create the plot
        fig = go.Figure()

        # Plot min-max range as a filled area
        fig.add_trace(go.Scatter(
            x=np.arange(max_length),  # X-axis (assuming index)
            y=max_spectrum,  # Upper bound of the min-max range
            mode='lines',
            line=dict(color='lightgray'),  # Light gray for max line
            fill=None,
            showlegend=False  # No legend for this trace
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(max_length),  # X-axis (assuming index)
            y=min_spectrum,  # Lower bound of the min-max range
            mode='lines',
            line=dict(color='lightgray'),  # Light gray for min line
            fill='tonexty',  # Fill the region between min and max
            fillcolor='lightgray',
            showlegend=False  # No legend for this trace
        ))

        # Plot the mean intensity as a separate line
        fig.add_trace(go.Scatter(
            x=np.arange(max_length),  # X-axis (assuming index)
            y=mean_spectrum,  # Mean intensity
            mode='lines',
            name='Mean Intensity',
            line=dict(width=3, color='blue')  # Customize line appearance
        ))

        # Plot each individual spectrum's intensity
        for i, spectrum in enumerate(padded_spectra):
            fig.add_trace(go.Scatter(
                y=spectrum,  # Intensity values
                mode='lines',  # Connect points with lines
                name=f'Spec.{i + 1}',  # Name for the legend
                line=dict(width=1, dash='dot'),  # Dotted line for individual spectra
                opacity=0.3  # Lower opacity for individual spectra
            ))

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Intensity",
            legend_title="Spectra",
            template='plotly',
            margin=dict(l=20, r=20, b=20, t=20, pad=0),
            autosize=True
        )

        return fig
    
    elif display_option == "gaussian":
        fig = go.Figure()

        # Plot each spectrum's intensity
        for i, spectrum in enumerate(selected_spectra):
            fig.add_trace(go.Scatter(
                y=spectrum,  # Intensity values
                mode='lines',  # Connect points with lines
                name=file_names[i],  # Name for the legend
                line=dict(width=2)  # Line width for visibility
            ))

        # Update layout for better visualization
        fig.update_layout(
            # title="Intensity vs. Index",
            xaxis_title="Index",
            yaxis_title="Intensity",
            legend_title="Spectra",
            template='plotly',  # You can change this to other templates if desired
            margin=dict(l=20, r=20, b=20, t=20, pad=0),
            autosize=True
        )

        return fig
   

def generate_scatter_plot(
    latent_vectors,
    bm_latent_vectors,
    n_components,
    cluster_selection=-1,
    clusters=None,
    cluster_names=None,
    bm_clusters=None,
    bm_cluster_names=None,
    label_selection=-2,
    labels=None,
    label_names=None,
    bm_label_selection=-2,
    bm_labels=None,
    bm_label_names=None,
    color_by="label",
):
    """
    Generate data for a scatter plot according to the provided selection options.
    """
 
    # Convert latent vectors to a numpy array and ensure correct types
    latent_vectors = np.array(latent_vectors, dtype=object)
    latent_vectors[:, :2] = latent_vectors[:, :2].astype(float)
    
    # Extract file names for hover information
    file_names = latent_vectors[:, 2].tolist()  # Assume the file names are in the 3rd column


    bm_latent_vectors = np.array(bm_latent_vectors, dtype=object)
    bm_latent_vectors[:, :2] = bm_latent_vectors[:, :2].astype(float)
    
    # Extract file names for hover information
    bm_file_names = bm_latent_vectors[:, 2].tolist()  # Assume the file names are in the 3rd column   

    # Initialize labels if not provided
    if labels is None:
        labels = np.full((latent_vectors.shape[0],), -1)
    if bm_labels is None:
        bm_labels = np.full((bm_latent_vectors.shape[0],), -1)

    # Determine values to color by
    if color_by == "cluster":
        vals = clusters
        vals_names = cluster_names
    else:
        vals = labels
        vals_names = {value: key for key, value in label_names.items()} if label_names is not None else {}
        vals_names[-1] = "Unlabeled"
        
    bm_vals = bm_clusters if color_by == "cluster" else bm_labels
    bm_vals_names = bm_cluster_names if color_by == "cluster" else {value: key for key, value in bm_label_names.items()} if bm_label_names is not None else {}
    bm_vals_names[-1] = "Unlabeled"

    # Create the scatter plot
    if n_components == 2:
        scatter_data = generate_scattergl_plot(
            latent_vectors[:, 0], 
            latent_vectors[:, 1],
            bm_latent_vectors[:, 0],
            bm_latent_vectors[:, 1],
            vals,
            bm_vals,
            vals_names, 
            bm_vals_names,
            file_names=file_names,
            bm_file_names=bm_file_names,
        )
    else:
        scatter_data = generate_scatter3d_plot(
            latent_vectors[:, 0],
            latent_vectors[:, 1],
            latent_vectors[:, 2],
            vals,
            vals_names,
            file_names=file_names
        )

    fig = go.Figure(scatter_data)
    fig.update_layout(
        dragmode="lasso",
        margin=dict(l=20, r=20, b=20, t=20, pad=0),
        legend=dict(tracegroupgap=20),
    )
    return fig

def generate_scattergl_plot(
    x_coords,
    y_coords,
    x_coords2,
    y_coords2,
    labels,
    bm_labels,
    label_to_string_map,
    bm_label_to_string_map,
    file_names,
    bm_file_names,
    show_legend=False,
    custom_indices=None,
):
    """
    Generate a scatter plot using Plotly's Scattergl for two-dimensional data.
    """
    
    unique_labels = set(labels)
    bm_unique_labels = set(bm_labels)
    
    traces = []
    # Create traces for the main data
    for label in unique_labels:
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]

        hover_text = [
            f"File: {file_names[i]}<br>X: {trace_x[j]:.2f}<br>Y: {trace_y[j]:.2f}"
            for j, i in enumerate(trace_indices)
        ]

        traces.append(
            go.Scattergl(
                x=trace_x,
                y=trace_y,
                customdata=np.array(hover_text).reshape(-1, 1),
                mode="markers+text",
                text = ['' for i in trace_indices],
                #text=[file_names[i] if file_names[i] in ['SigScan4200.txt', 'SigScan4200_diff.txt', 'SigScan4200_same.txt'] else '' for i in trace_indices],
                textposition="top center",
                name=str(label_to_string_map[label]),
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    # Create traces for the background model data
    for label in bm_unique_labels:
        trace_indices = [i for i, l in enumerate(bm_labels) if l == label]
        trace_x = [x_coords2[i] for i in trace_indices]
        trace_y = [y_coords2[i] for i in trace_indices]

        hover_text = [
            f"File: {bm_file_names[i]}<br>X: {trace_x[j]:.2f}<br>Y: {trace_y[j]:.2f}"
            for j, i in enumerate(trace_indices)
        ]

        traces.append(
            go.Scattergl(
                x=trace_x,
                y=trace_y,
                customdata=np.array(hover_text).reshape(-1, 1),
                mode="markers+text",
                text = ['' for i in trace_indices],
                #text=[bm_file_names[i] if bm_file_names[i] in ['SigScan4200.txt', 'SigScan4200_diff.txt', 'SigScan4200_same.txt'] else '' for i in trace_indices],
                textposition="top center",
                name=str(bm_label_to_string_map[label]),
                hovertemplate="%{customdata}<extra></extra>",
            )
        )


    fig = go.Figure(data=traces)
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(255, 255, 255, 0.9)",
                orientation="h",
            )
        )
    return fig


def generate_scatter3d_plot(
    x_coords,
    y_coords,
    z_coords,
    labels,
    label_to_string_map,
    file_names,
    show_legend=False,
    custom_indices=None,
):
    unique_labels = set(labels)
    traces = []
    for label in unique_labels:
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]
        trace_z = [z_coords[i] for i in trace_indices]

        if custom_indices is not None:
            trace_custom_indices = [custom_indices[i] for i in trace_indices]
        else:
            trace_custom_indices = trace_indices

        hover_text = [
            f"File: {file_names[i]}<br>X: {trace_x[j]:.2f}<br>Y: {trace_y[j]:.2f}<br>Z: {trace_z[j]:.2f}"
            for j, i in enumerate(trace_indices)
        ]

        traces.append(
            go.Scatter3d(
                x=trace_x,
                y=trace_y,
                z=trace_z,
                customdata=np.array(hover_text).reshape(-1, 1),
                mode="markers+text",
                text = ['' for i in trace_indices],
                #text=[file_names[i] if file_names[i] in ['SigScan4200.txt', 'SigScan4200_diff.txt', 'SigScan4200_same.txt'] else '' for i in trace_indices],
                textposition="top center",
                name=str(label_to_string_map[label]),
                marker=dict(size=3),
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

   
    fig = go.Figure(data=traces)
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(255, 255, 255, 0.9)",
                orientation="h",
            )
        )
    return fig

 