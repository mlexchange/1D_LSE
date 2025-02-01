# Import necessary modules
from dash import Input, Output, State, callback, no_update
from utils.data_utils import load_latent_vectors, load_spectra_data, load_benchmark_filenames, BASE_DATA, BASE
from dash.exceptions import PreventUpdate 
import os
import pandas as pd
import pyroved as pv
import torch
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import base64
from io import StringIO



MIN_PEAKS = 15
LEN_INTENSITY = 500
base = BASE

# Define a Gaussian peak-fitting function
def fit_gaussian_peak(i1):
    
    i1 = np.array(i1)
    
    def peakfunc(x, A, mu, sigma):
        """Define the Gaussian peak function"""
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def gaussian_sum(x, *params):
        """Sum of multiple Gaussian functions"""
        n = len(params) // 2  # Number of Gaussians
        A_values = params[:n]  # Amplitudes
        sigma_values = params[n:]  # Standard deviations
        total = np.zeros_like(x)
        for i in range(n):
            total += peakfunc(x, A_values[i], mu_values[i], sigma_values[i])
        return total

 
    def normalize_intensity(intensity):
        """Normalize intensity values between 0 and 1"""
        intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        x_values = np.linspace(0, 1, len(intensity))   
        return x_values, intensity

    
    x, y = normalize_intensity(i1)
    # Find peak positions (mu_values) from the data using find_peaks
    peaks, _ = find_peaks(y, height=0.2, distance=10)  # Adjust height and distance to control detection

    n = min(5,len(peaks))
    # Ensure there are exactly `n` peaks, otherwise adjust
    if len(peaks) < n:
        print(f"Warning: Detected fewer peaks ({len(peaks)}) than expected ({n}). Filling remaining peaks.")
        mu_values = list(x[peaks]) + list(np.linspace(0.1, 0.9, n - len(peaks)))  # Fill with some default values
    elif len(peaks) > n:
        print(f"Warning: Detected more peaks ({len(peaks)}) than expected ({n}). Trimming extra peaks.")
        mu_values = list(x[peaks[:n]])  # Trim to exactly `n` peaks
    else:
        mu_values = x[peaks]  # Exactly `n` peaks detected

    # Initial guesses for A and sigma
    initial_guess_A = [y[peak] for peak in peaks[:n]]  # A values based on peak heights
    initial_guess_sigma = [0.05] * n  # Initial sigma guess (you can refine this)
    initial_guess = initial_guess_A + initial_guess_sigma

    # Perform the curve fitting to find optimal A_values and sigma_values
    popt, pcov = curve_fit(gaussian_sum, x, y, p0=initial_guess,maxfev=1000000)

    # Extract the fitted A_values and sigma_values
    fitted_A_values = popt[:n]
    fitted_sigma_values = popt[n:]

    # Print the fitted parameters
    print("Fitted parameters:")
    for i in range(n):
        print(f"Gaussian {i+1}: A={fitted_A_values[i]:.2f}, mu={mu_values[i]:.2f}, sigma={fitted_sigma_values[i]:.2f}")
    gauss_fit = gaussian_sum(x, *popt)
    return gauss_fit, fitted_A_values,mu_values,fitted_sigma_values

def peakfunc2(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_sum2(x, a, m, s):
    
    A_values = a  # Amplitudes
    sigma_values = s  # Standard deviations
    mu_values = m
    total = np.zeros_like(x)
    for i in range(len(mu_values)):
        total += peakfunc2(x, A_values[i], mu_values[i], sigma_values[i])
    return total


file_counts = {}
@callback(
    Output("data-store", "data"),
    Output("data-store2", "data"),
    Output("latent-vectors", "data"),
    Output("latent-vectors2", "data"), 
    Output("spectrum-dropdown", "options"),
    Input("generate-data-button", "n_clicks"),
)
def update_data(n_clicks):
    if n_clicks:
        data, bm_data = load_spectra_data()
        latent_vectors, bm_latent_vectors = load_latent_vectors()
        filenames = load_benchmark_filenames()
        dropdown_options = [{'label': f, 'value': f} for f in filenames]
        return data, bm_data, latent_vectors, bm_latent_vectors, dropdown_options
    return no_update, no_update, no_update, no_update, no_update

@callback(
    [Output("energy-store", "data"), Output("intensity-store", "data")],
    [Input("spectrum-dropdown", "value")]
)
def load_selected_file_data(selected_filename):
    data, benchmark_intensity = load_spectra_data()
    filenames = load_benchmark_filenames()
    benchmark_file_path = os.path.join(BASE_DATA, 'benchmark_data_vae.parquet')
    benchmark_df = pd.read_parquet(benchmark_file_path)

    if selected_filename in filenames:
        file_data = benchmark_df[benchmark_df['file_name'] == selected_filename]
        if not file_data.empty:
            energy = file_data['energy'].tolist()[0]
            intensity = file_data['intensity'].tolist()[0]
            return energy, intensity
    
    return no_update, no_update

@callback(
    Output("append-status", "data"),
    Input("append-data-button", "n_clicks"),
    Input("energy-store", "data"),
    Input("intensity-store", "data"),
    State("spectrum-dropdown", "value"),
    State("amplitude-shift", "value"),
    State("mean-shift", "value"),
    State("sigma-shift", "value"),
    State("noise-shift", "value")
)
def append_selected_data(n_clicks, energy, intensity, filename, amplitude_shift=0.0, mean_shift=0.0, sigma_shift=0.0, noise_shift=0.0):
    if not (n_clicks and energy and intensity and filename):
        raise PreventUpdate("No data to append or button not clicked.")
    
    intensity_array = np.array(intensity)
    
    # Apply amplitude shift
    if amplitude_shift != 0:
        intensity_array *= amplitude_shift  # Amplify intensity

    # Apply mean shift
    if mean_shift != 0:
        x_original = np.arange(len(intensity_array))
        x_shifted = x_original - mean_shift
        x_shifted = np.clip(x_shifted, 0, len(intensity_array) - 1)  # Ensure no out-of-bounds indices
        interpolator = interp1d(x_original, intensity_array, kind='linear', fill_value="extrapolate")
        intensity_array = interpolator(x_shifted)
    
    # Apply sigma shift
    if sigma_shift != 0:
        
        g,a,m,s = fit_gaussian_peak(intensity_array)
        
     
        s[0] = s[0] + sigma_shift 
        x = np.linspace(0,1,len(g))
        
        intensity_array = gaussian_sum2(x, a, m, s)

    # Add noise
    if noise_shift != 0:
        noise_std = noise_shift * np.mean(np.abs(intensity_array))  # Scale noise based on signal amplitude
        noise = np.random.normal(0, noise_std, intensity_array.shape)
        intensity_array += noise

    # Normalize intensity
    unnorm = intensity_array
    unnorm_list = list(unnorm)
    intensity_array = (intensity_array - np.min(intensity_array)) / (np.max(intensity_array) - np.min(intensity_array))
    intensity = list(intensity_array)
    
    # Load model and compute latent vectors
    data_array = np.array([list(unnorm)])

    sample_tensor = torch.tensor(data_array, dtype=torch.float32)
    svae = load_model()
    
    with torch.no_grad():
        z_mean, z_sd = svae.encode(sample_tensor)

    # Assign unique filename
    if filename in file_counts:
        file_counts[filename] += 1
    else:
        file_counts[filename] = 1
    unique_filename = f"{filename}_{file_counts[filename]}"

    # Save data
    # df_temp = pd.DataFrame({'file_name': [unique_filename], 'energy': [energy], 'intensity': [intensity], 'cluster': [0]})
    df_temp = pd.DataFrame({'file_name': [unique_filename], 'energy': [energy], 'intensity': [unnorm_list], 'cluster': [0]})
    df_latent = pd.DataFrame({'0': z_mean[:, -1].numpy(), '1': z_mean[:, -2].numpy(), 'file_name': [unique_filename]})

    data_file_path = os.path.join(os.getcwd(), 'data', 'relevant_data1.parquet')
    latent_file_path = os.path.join(os.getcwd(), 'data', 'latent_vectors_relevant_data1.parquet')
    os.makedirs(os.path.dirname(data_file_path), exist_ok=True)

    if os.path.exists(data_file_path):
        existing_data = pd.read_parquet(data_file_path)
        df_data = pd.concat([existing_data, df_temp], ignore_index=True)
    else:
        df_data = df_temp
    df_data.to_parquet(data_file_path, index=False)

    if os.path.exists(latent_file_path):
        existing_latent = pd.read_parquet(latent_file_path)
        df_latent = pd.concat([existing_latent, df_latent], ignore_index=True)
    df_latent.to_parquet(latent_file_path, index=False)

    return f"Data for {unique_filename} appended successfully."


@callback(
    Output("browse-data-upload", "children"),  
    Input("browse-data-upload", "contents"),
    State("browse-data-upload", "filename"),
)

def process_selected_file(contents, filename):
    if contents is None:
        raise PreventUpdate  # Do nothing if no file is uploaded

    # Decode file content
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')

    
    df = pd.read_csv(StringIO(decoded), sep="\t")  # Adjust delimiter if needed

    # Ensure required columns are present
    if 'intensity' not in df.columns or 'energy' not in df.columns:
        df.columns = ['energy', 'intensity'] 
            
    energy = df['energy'].tolist()
    intensity_array = df['intensity'].tolist()
    
    x_shifted = np.linspace(np.min(energy), np.max(energy), LEN_INTENSITY)  # Ensure no out-of-bounds indices
    interpolator = interp1d(energy, intensity_array, kind='linear', fill_value="extrapolate")
    intensity_array = interpolator(x_shifted)
    
    
    energy = x_shifted
    
    # Normalize intensity
    unnorm = intensity_array
    unnorm_list = list(unnorm)
    intensity_array = (intensity_array - np.min(intensity_array)) / (np.max(intensity_array) - np.min(intensity_array))
    intensity = list(intensity_array)
    
    # Load model and compute latent vectors
    data_array = np.array([list(intensity_array)])
    sample_tensor = torch.tensor(data_array, dtype=torch.float32)
    svae = load_model()
    
    with torch.no_grad():
        z_mean, z_sd = svae.encode(sample_tensor)

    print("file_counts",file_counts)
    # Assign unique filename
    if filename in file_counts:
        file_counts[filename] += 1
    else:
        file_counts[filename] = 1
    unique_filename = f"{filename}_{file_counts[filename]}"

    # Save data
    df_temp = pd.DataFrame({'file_name': [unique_filename], 'energy': [energy], 'intensity': [unnorm_list], 'cluster': [0]})
    df_latent = pd.DataFrame({'0': z_mean[:, -1].numpy(), '1': z_mean[:, -2].numpy(), 'file_name': [unique_filename]})

    data_file_path = os.path.join(base, 'data', 'relevant_data1.parquet')
    latent_file_path = os.path.join(base, 'data', 'latent_vectors_relevant_data1.parquet')

    if os.path.exists(data_file_path):
        existing_data = pd.read_parquet(data_file_path)
        df_data = pd.concat([existing_data, df_temp], ignore_index=True)
    else:
        df_data = df_temp
        
    print(df_data)
    df_data.to_parquet(data_file_path, index=False)

    if os.path.exists(latent_file_path):
        existing_latent = pd.read_parquet(latent_file_path)
        df_latent = pd.concat([existing_latent, df_latent], ignore_index=True)
    df_latent.to_parquet(latent_file_path, index=False)
    print(os.getcwd())
    print("data written successfully")

    return f"Data for {unique_filename} appended successfully."




def load_model():
    in_dim = (500,)
    svae = pv.models.iVAE(in_dim, latent_dim=2, invariances=['t'], dx_prior=0.3)
    svae.load_state_dict(torch.load(os.path.join(BASE_DATA, 'svae_model_100.pth')))
    return svae
