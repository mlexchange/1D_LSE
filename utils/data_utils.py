import numpy as np
import pandas as pd
import os

# Directory where the data files are stored
#READ_DIR = '/Users/monikachoudhary/Documents/mlex_spectral_viz_benchmarking4/data/'
BASE_DATA = '/Users/monikachoudhary/Documents/mlex_spectral_viz_benchmarking4/data/'
BASE = '/Users/monikachoudhary/Documents/mlex_spectral_viz_benchmarking4/'

def load_spectra_data():
    """
    Loads the spectra data from a parquet file.

    Returns:
        list: A list containing the intensity values of the spectra.
    """
    # Path to the data.parquet file
    print('loading spectra data')
    spectra_file_path = os.path.join(BASE_DATA, 'relevant_data1.parquet')
    benchmark_file_path = os.path.join(BASE_DATA, 'benchmark_data_vae.parquet')
    # Load the data into a Pandas DataFrame
    df_spectra = pd.read_parquet(spectra_file_path)
    benchmark_spectra = pd.read_parquet(benchmark_file_path)
    
    # Extract the 'intensity' column as a list
    data = df_spectra['intensity'].tolist()
    benchmark = benchmark_spectra['intensity'].tolist()
    
    return data,benchmark

def load_benchmark_filenames():
    """
    Load the benchmark data and extract filenames.
    """
    print('loading benchmark filenames')
    # Example of loading data - adjust as per your actual data format
    benchmark_file_path = os.path.join(BASE_DATA, 'benchmark_data_vae.parquet')
    df = pd.read_parquet(benchmark_file_path)
    
    # Assuming the filenames are in a column 'file_name'
    filenames = df['file_name'].unique().tolist()
    return filenames


def load_latent_vectors():
    """
    Loads the latent vectors from a parquet file.

    Returns:
        np.ndarray: A NumPy array of latent vectors.
    """
    print('loading latent vectors')
    # Path to the latent_vectors.parquet file
    lv_file_path = os.path.join(BASE_DATA, 'latent_vectors_relevant_data1.parquet')
    bm_file_path = os.path.join(BASE_DATA, 'benchmark_latent_vectors_vae.parquet')
    # Load the latent vectors into a Pandas DataFrame
    df_lv = pd.read_parquet(lv_file_path)
    bm_lv = pd.read_parquet(bm_file_path)
    print(df_lv)

    # Convert the DataFrame to a NumPy array
    latent_vectors = df_lv.to_numpy()
    bm_latent_vectors = bm_lv.to_numpy()
    
    return latent_vectors,bm_latent_vectors
