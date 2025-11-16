import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
import tempfile
import os
import shutil
import requests # Import requests library

# --- Constants ---
# IMPORTANT: Replace this with your actual Google Drive direct download URL
# For example: "https://docs.google.com/uc?export=download&id=YOUR_FILE_ID"
MODEL_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=134y3Q2sfXoBxDiwhZ4wIbXPSO-vHl8e8" # Updated with your model's direct download URL

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
def load_model_from_url(model_url):
    """
    Downloads a model from a URL to a temporary file and then loads it.
    """
    st.info("Downloading model...")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors

        # Use NamedTemporaryFile to create a temporary file to store the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        st.success(f"Model downloaded to {tmp_path}")
        model = tf.keras.models.load_model(tmp_path)
        st.success("Model loaded successfully!")
        # Optional: Delete temporary file after loading if memory permits and model is cached.
        # For st.cache_resource, the model itself is cached, so the file can be deleted.
        os.remove(tmp_path)
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download model: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model from downloaded file: {e}")
        return None

# --- Streamlit Application ---
st.title('3D CNN Cancer Classification')
st.write("Upload SUV and CTres NIfTI files to get a cancer classification prediction.")

# Load the model once when the app starts, using the download logic
model = load_model_from_url(MODEL_DOWNLOAD_URL)

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
