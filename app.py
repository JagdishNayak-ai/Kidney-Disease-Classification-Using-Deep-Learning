import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Kidney Diagnostic Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_kidney_model.keras')
LAST_CONV_LAYER_NAME = 'out_relu'


@st.cache_resource
def load_kidney_model():
    return tf.keras.models.load_model(MODEL_PATH)


def is_valid_ct_scan(img_np):
    """
    Input Guardrail: Validates whether the uploaded image is a single-crop CT scan
    or an Out-of-Distribution (OOD) document/ultrasound report printout.
    """
    # 1. Check for document/paper background (high ratio of pure white background pixels)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    white_pixel_ratio = np.sum(gray > 240) / gray.size
    
    if white_pixel_ratio > 0.30:
        return False, "Detected document/paper background. Please upload an isolated CT scan slice."

    # 2. Check for multi-panel aspect ratio (standard CT slices are roughly 1:1 square crops)
    h, w, _ = img_np.shape
    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 1.4:
        return False, "Detected multi-panel page layout. Please upload an individual CT scan crop."

    return True, "Valid CT Scan"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs[0] if isinstance(model.inputs, list) else model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        pred_score = predictions[0][0]
        
        # Dynamically track target score for predicted class (Tumor vs Normal)
        target_score = pred_score if pred_score >= 0.5 else (1.0 - pred_score)

    # Compute gradients with respect to target feature maps
    grads = tape.gradient(target_score, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU and safely normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


# ==========================================
# HEADER & SIDEBAR
# ==========================================
st.title("🩺 AI-Powered Kidney Disease Diagnostic Portal")
st.markdown("Computer-Assisted Abdominal CT Scan Analysis with MobileNetV2 & Explainable AI (XAI)")

try:
    model = load_kidney_model()
    st.sidebar.success("Model Status: Online (MobileNetV2)")
except Exception as e:
    st.sidebar.error("Model Loading Error")
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("⚙️ Diagnostics Setup")
uploaded_file = st.sidebar.file_uploader("Upload Abdominal CT Scan...", type=["jpg", "jpeg", "png"])
alpha_slider = st.sidebar.slider("Grad-CAM Overlay Intensity", min_value=0.1, max_value=1.0, value=0.4, step=0.05)


# ==========================================
# MAIN INTERFACE (TABS)
# ==========================================
tab_diag, tab_pipeline, tab_metrics, tab_about = st.tabs([
    "📊 Clinical Dashboard", 
    "🔬 Intermediate CNN Feature Pipeline", 
    "📈 Model Performance",
    "ℹ️ System Info"
])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(pil_img)
    
    # --- INPUT VALIDATION GUARDRAIL ---
    is_valid, error_msg = is_valid_ct_scan(img_np)
    
    if not is_valid:
        with tab_diag:
            st.error("⚠️ Invalid Image Type Detected")
            st.warning(f"Error: {error_msg}")
            st.info("💡 Note: This portal is trained strictly on single 2D Abdominal CT Scans. Uploading ultrasound reports or paper documents results in Out-of-Distribution errors.")
        st.stop()
    # ----------------------------------

    img_resized = cv2.resize(img_np, (224, 224))
    img_uint8 = img_resized.copy()
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_normalized, axis=0)

    # Model Inference
    start_time = time.time()
    prediction_prob = float(model.predict(img_tensor, verbose=0)[0][0])
    inference_time = (time.time() - start_time) * 1000

    label = "TUMOR DETECTED" if prediction_prob >= 0.5 else "NORMAL (HEALTHY)"
    confidence = prediction_prob if prediction_prob >= 0.5 else (1 - prediction_prob)

    # Grad-CAM Heatmap
    heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV_LAYER_NAME)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_uint8, 1 - alpha_slider, jet_heatmap, alpha_slider, 0)

    # ------------------------------------------
    # TAB 1: CLINICAL DASHBOARD
    # ------------------------------------------
    with tab_diag:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            if "TUMOR" in label:
                st.metric("Diagnostic Result", label, delta="Abnormal Mass Detected", delta_color="inverse")
            else:
                st.metric("Diagnostic Result", label, delta="Clear Scan", delta_color="normal")
        with col_m2:
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")
        with col_m3:
            st.metric("Inference Speed", f"{inference_time:.1f} ms")

        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 1. Original CT Scan")
            st.image(img_resized, use_container_width=True)
        with c2:
            st.markdown("##### 2. Grad-CAM Heatmap")
            st.image(heatmap_resized, clamp=True, use_container_width=True)
        with c3:
            st.markdown("##### 3. Diagnostic Overlay")
            st.image(superimposed_img, use_container_width=True)

    # ------------------------------------------
    # TAB 2: INTERMEDIATE FEATURE PIPELINE
    # ------------------------------------------
    with tab_pipeline:
        st.header("Step-by-Step Deep Learning Feature Extraction")
        st.markdown("Visualizing how the CT scan is processed through structural filters and neural layers.")

        # Edge Detection Processing
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_img, 100, 200)

        # Dynamic ROI Focus based on Peak Heatmap Activation
        max_heat = np.max(heatmap_uint8)
        threshold_val = max(int(max_heat * 0.75), 180)
        
        _, thresh = cv2.threshold(heatmap_uint8, threshold_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_img = img_resized.copy()
        
        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
            if valid_contours:
                largest_cnt = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_cnt)
                
                pad = 10
                x_min = max(0, x - pad)
                y_min = max(0, y - pad)
                x_max = min(224, x + w + pad)
                y_max = min(224, y + h + pad)
                
                box_color = (255, 0, 0) if "TUMOR" in label else (0, 255, 0)
                cv2.rectangle(roi_img, (x_min, y_min), (x_max, y_max), box_color, 2)

        # Single-line headers for grid alignment
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown("##### 1. Input Image")
            st.image(img_resized, use_container_width=True)
        with p2:
            st.markdown("##### 2. Edges")
            st.image(edges, use_container_width=True)
        with p3:
            st.markdown("##### 3. Heatmap")
            st.image(heatmap_resized, clamp=True, use_container_width=True)
        with p4:
            st.markdown("##### 4. ROI Focus")
            st.image(roi_img, use_container_width=True)

    # ------------------------------------------
    # TAB 3: MODEL METRICS
    # ------------------------------------------
    with tab_metrics:
        st.header("Model Performance Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test Accuracy", "99.59%")
        k2.metric("Backbone Architecture", "MobileNetV2")
        k3.metric("Training Epochs", "15")
        k4.metric("Dataset Size", "~7,360 Images")
        
        st.markdown("---")
        st.markdown("### Model Architecture Details")
        st.markdown("""
        * **Backbone Feature Extractor:** Pre-trained **MobileNetV2** utilizing Depthwise Separable Convolutions to extract multi-scale spatial features[cite: 1, 3].
        * **Target Layer for XAI:** `out_relu` (Final 2D convolutional activation layer producing $7 \\times 7 \\times 1280$ feature maps)[cite: 1, 3].
        * **Classification Head:** `GlobalAveragePooling2D` $\\rightarrow$ `BatchNormalization` $\\rightarrow$ `Dense(512, ReLU)` $\\rightarrow$ `Dropout(0.5)` $\\rightarrow$ `Dense(1, Sigmoid)`.
        """)

    # ------------------------------------------
    # TAB 4: SYSTEM INFO
    # ------------------------------------------
    with tab_about:
        st.header("About the Diagnostic Portal")
        st.write("This diagnostic assistant system was developed to evaluate transfer learning and Explainable AI (XAI) techniques for detecting kidney disease from 2D abdominal CT scans.")

else:
    with tab_diag:
        st.info("👈 Please upload a CT scan file from the left sidebar to generate diagnostics.")