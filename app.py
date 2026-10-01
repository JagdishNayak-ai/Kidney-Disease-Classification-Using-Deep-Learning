import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Kidney Disease Diagnostic Assistant",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_kidney_model.keras')
LAST_CONV_LAYER_NAME = 'out_relu'


@st.cache_resource
def load_kidney_model():
    """Cache the model in memory so it doesn't reload on every interaction."""
    return tf.keras.models.load_model(MODEL_PATH)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generates Grad-CAM activation heatmap."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs[0] if isinstance(model.inputs, list) else model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        class_channel = predictions[0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


# UI Header
st.title("🩺 AI-Powered Kidney Disease Classification")
st.markdown("Upload a abdominal CT scan image to predict **Tumor** vs. **Normal** and view **Grad-CAM visual explainability**.")
st.divider()

# Model Loading
try:
    model = load_kidney_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar File Upload
st.sidebar.header("Upload Scan")
uploaded_file = st.sidebar.file_uploader("Choose a CT Scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image File
    pil_img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(pil_img)
    
    # Resize and Preprocess
    img_resized = cv2.resize(img_np, (224, 224))
    img_uint8 = img_resized.copy()
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_normalized, axis=0)

    # Perform Prediction
    with st.spinner("Analyzing CT scan..."):
        prediction_prob = float(model.predict(img_tensor, verbose=0)[0][0])
        label = "TUMOR" if prediction_prob >= 0.5 else "NORMAL"
        confidence = prediction_prob if prediction_prob >= 0.5 else (1 - prediction_prob)

    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV_LAYER_NAME)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Color map & Superimpose
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_uint8, 0.6, jet_heatmap, 0.4, 0)

    # Display Metrics & Results
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        if label == "TUMOR":
            st.error(f"### Diagnosis: {label}")
        else:
            st.success(f"### Diagnosis: {label}")
    with col_metric2:
        st.metric(label="Model Confidence Level", value=f"{confidence * 100:.2f}%")

    st.divider()

    # Display Images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Original CT Scan")
        st.image(img_resized, use_container_width=True)

    with col2:
        st.subheader("2. Grad-CAM Heatmap")
        st.image(heatmap_resized, clamp=True, use_container_width=True)

    with col3:
        st.subheader("3. Explainable Overlay")
        st.image(superimposed_img, use_container_width=True)

else:
    st.info(" Please upload a CT scan file from the sidebar to begin analysis.")