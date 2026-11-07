from io import BytesIO
import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Load environment variables from .env file
load_dotenv()

# 2. Initialize Gemini Client
client = genai.Client()

# Page configuration
st.set_page_config(
    page_title="Kidney Diagnostic Agent", page_icon="🩺", layout="centered"
)

st.title("🩺 Kidney Disease Classification & Diagnostic Agent")
st.write(
    "Upload a kidney CT scan image for AI-powered classification and clinical"
    " summary."
)


# 3. Load Model (Cached using shared master environment)
@st.cache_resource
def load_classification_model():
  model_path = os.path.join("models", "best_kidney_model.keras")
  if os.path.exists(model_path):
    try:
      return tf.keras.models.load_model(model_path)
    except Exception as e:
      st.error(f"Error loading model: {e}")
      return None
  return None


model = load_classification_model()

# 4. File Uploader Interface
uploaded_file = st.file_uploader(
    "Choose a kidney CT scan image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Open and display original CT scan image
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Scan", use_container_width=True)

  if st.button("Run Diagnostic Analysis"):
    if model is None:
      st.warning(
          "⚠️ Model file (`best_kidney_model.keras`) not found in the `models/`"
          " folder. Running Gemini GenAI fallback analysis..."
      )
    else:
      with st.spinner("Processing image through TensorFlow model..."):
        # Preprocess image to match MobileNet training pipeline (224x224, normalized [0, 1])
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)  # Add batch axis
        img_array = img_array / 255.0  # Rescaling matching dataset.py

        # Make Prediction (Sigmoid single-node output)
        prediction_prob = float(model.predict(img_array, verbose=0)[0][0])

        # 🔍 Print this exact number to your terminal window
        print(f"--- DEBUG: Raw Model Probability for Tumor = {prediction_prob:.4f} ---")

        if prediction_prob >= 0.5:
          predicted_class = "Tumor"
          confidence = prediction_prob * 100
        else:
          predicted_class = "Normal"
          confidence = (1 - prediction_prob) * 100

        st.success(
            f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)"
        )

    # 5. Generate GenAI Clinical Context via Gemini with Fallback
    with st.spinner("Consulting Gemini Clinical Assistant..."):
      prompt = (
          "You are an expert AI medical assistant. A patient kidney CT scan "
          "has been uploaded. Provide a clear, simple summary in everyday language "
          "using bullet points. Structure your response under these exact headings:\n\n"
          "• **What We See on the Scan:** (Describe the findings simply)\n"
          "• **What This Means:** (Explain the condition in basic terms)\n"
          "• **Recommended Next Steps:** (List simple actions like consulting a doctor or getting lab work done)\n\n"
          "Keep the language straightforward and avoid complex medical jargon where possible."
      )

      # List of models to try in order of preference
      models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash"]
      response = None
      success = False

      for model_name in models_to_try:
        try:
          response = client.models.generate_content(
              model=model_name, contents=[prompt, image]
          )
          success = True
          break  # Exit loop if successful
        except Exception:
          continue  # Try next model if current one fails

      if success and response:
        st.subheader("📋 GenAI Clinical Summary & Insights")
        st.write(response.text)
      else:
        st.error(
            "⚠️ Gemini models are currently experiencing high demand. Please"
            " wait a moment and click 'Run Diagnostic Analysis' again."
        )