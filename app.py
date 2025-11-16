import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
import tempfile
import os
import shutil
# import requests # No longer needed if using gdown for primary download
import re # Still useful for general regex if needed elsewhere, or can remove if not
import gdown # Import gdown library

# --- Constants ---
# IMPORTANT: Replace this with your actual Google Drive direct download link
MODEL_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=134y3Q2sfXoBxDiwhZ4wIbXPSO-vHl8e8"
# Extract File ID for gdown
# The file ID is the part after 'id=' in the MODEL_DOWNLOAD_URL
MODEL_FILE_ID = MODEL_DOWNLOAD_URL.split('id=')[-1]

OPTIMAL_THRESHOLD = 0.2
TARGET_SHAPE = (64, 64, 64)

# --- Functions ---

@st.cache_data
def preprocess_single_nii(nii_file_path, target_shape=TARGET_SHAPE, normalization_method='min_max'):
    """
    Loads, resamples, and normalizes a single NIfTI image.
    """
    img = nib.load(nii_file_path)
    img_data = img.get_fdata()

    # Resampling
    if img_data.shape != target_shape:
        zoom_factors = [ts / cs for ts, cs in zip(target_shape, img_data.shape)]
        img_data = zoom(img_data, zoom_factors, order=1) # order=1 for linear interpolation

    # Normalization
    if normalization_method == 'min_max':
        min_val = img_data.min()
        max_val = img_data.max()
        if (max_val - min_val) != 0:
            img_data = (img_data - min_val) / (max_val - min_val)
        else: # Handle case where all values are the same
            img_data = np.zeros_like(img_data)
    elif normalization_method == 'z_score':
        mean_val = img_data.mean()
        std_val = img_data.std()
        if std_val != 0:
            img_data = (img_data - mean_val) / std_val
        else: # Handle case where standard deviation is zero
            img_data = np.zeros_like(img_data)
    else:
        raise ValueError("Invalid normalization_method. Choose 'min_max' or 'z_score'.")

    return img_data

@st.cache_resource
def load_model_from_url(model_file_id):
    """
    Downloads a model from Google Drive using gdown and then loads it.
    """
    st.info("Attempting to download model from Google Drive...")
    tmp_path = None # Initialize tmp_path outside try block

    try:
        # Create a temporary file path for the downloaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            tmp_path = tmp_file.name

        # Use gdown to download the file directly
        gdown.download(id=model_file_id, output=tmp_path, quiet=False, fuzzy=False)

        st.success(f"Model downloaded to {tmp_path}")
        file_size_bytes = os.path.getsize(tmp_path)
        st.info(f"Downloaded file size: {file_size_bytes / (1024 * 1024):.2f} MB")
        
        # Add a check for file existence and content before loading
        if not os.path.exists(tmp_path) or file_size_bytes == 0:
            st.error(f"Error: Downloaded file does not exist or is empty at {tmp_path}")
            return None

        # Try to load the model
        model = tf.keras.models.load_model(tmp_path)
        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Error loading model from downloaded file: {e}")
        st.error(f"Attempted to download/load from Google Drive ID: {model_file_id}")
        if tmp_path and os.path.exists(tmp_path):
            file_size_bytes = os.path.getsize(tmp_path)
            st.error(f"File exists at {tmp_path} with size {file_size_bytes / (1024 * 1024):.2f} MB")
            # Optional: print first few lines of the file to check for HTML content if it's small
            if file_size_bytes > 0 and file_size_bytes < 1024 * 1024: # Check if it's less than 1MB to avoid printing huge files
                try:
                    with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_lines = f.read(500) # Read first 500 characters
                        if "<html" in first_lines.lower() or "<!doctype html>" in first_lines.lower():
                            st.error("WARNING: Downloaded file appears to be an HTML page, not a Keras model. "
                                     "This often means Google Drive required confirmation or a different download method.")
                        else:
                            st.info(f"First 500 chars of file: {first_lines}")
                except Exception as file_read_e:
                    st.error(f"Could not read temporary file for inspection: {file_read_e}")
        return None
    finally:
        # Clean up temporary file if it was created and delete=False
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Streamlit Application ---
st.title('3D CNN Cancer Classification')
st.write("Upload SUV and CTres NIfTI files to get a cancer classification prediction.")

# Load the model once when the app starts, using the download logic
model = load_model_from_url(MODEL_FILE_ID) # Pass the file ID to the function

if model is None:
    st.stop() # Stop the app if model loading failed

# File uploaders
uploaded_suv_file = st.file_uploader('Upload SUV NIfTI file', type=['nii', 'nii.gz'])
uploaded_ctres_file = st.file_uploader('Upload CTres NIfTI file', type=['nii', 'nii.gz'])

# Prediction button
if st.button('Predict'):
    if uploaded_suv_file is None or uploaded_ctres_file is None:
        st.error('Please upload both SUV and CTres NIfTI files.')
    else:
        with st.spinner('Preprocessing and predicting...'):
            try:
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_suv:
                    shutil.copyfileobj(uploaded_suv_file, tmp_suv)
                    suv_path = tmp_suv.name
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_ctres:
                    shutil.copyfileobj(uploaded_ctres_file, tmp_ctres)
                    ctres_path = tmp_ctres.name

                # Preprocess files
                suv_preprocessed = preprocess_single_nii(suv_path)
                ctres_preprocessed = preprocess_single_nii(ctres_path)

                # Create multi-channel input
                if suv_preprocessed.shape != TARGET_SHAPE or ctres_preprocessed.shape != TARGET_SHAPE:
                    raise ValueError(f"Preprocessing failed to return target shape {TARGET_SHAPE}")

                multi_channel_input = np.stack([suv_preprocessed, ctres_preprocessed], axis=-1)
                # Expand dimensions for batch size (model expects (1, D, H, W, C))
                multi_channel_input_batch = np.expand_dims(multi_channel_input, axis=0)

                # Make prediction
                prediction_probability = model.predict(multi_channel_input_batch)[0][0]

                # Apply optimal threshold
                # Model outputs P(Positive). If P(Positive) > threshold, it's 'Positive'.
                predicted_label_numerical = 0 if prediction_probability > OPTIMAL_THRESHOLD else 1
                predicted_class_name = 'Positive' if predicted_label_numerical == 0 else 'Negative'

                st.success(f'Predicted Classification: {predicted_class_name}')
                st.write(f'Probability of Positive: {prediction_probability:.4f} (using threshold {OPTIMAL_THRESHOLD:.1f})')

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
            finally:
                # Clean up temporary files
                if 'suv_path' in locals() and os.path.exists(suv_path):
                    os.remove(suv_path)
                if 'ctres_path' in locals() and os.path.exists(ctres_path):
                    os.remove(ctres_path)
