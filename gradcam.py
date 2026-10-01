import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_CATEGORY = 'Tumor'  # Options: 'Normal' or 'Tumor'
IMAGE_INDEX = 0            # Change index (0, 1, 2...) to pick different images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_kidney_model.keras')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Keras MobileNetV2 final activation layer name
LAST_CONV_LAYER_NAME = 'out_relu'


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for a given image array and model.
    """
    # Create sub-model mapping input layer directly to conv output and final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs[0] if isinstance(model.inputs, list) else model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        class_channel = predictions[0]

    # Compute gradients of prediction score w.r.t the feature map output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of mean intensity of gradients over specific feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def generate_gradcam():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    category_folder = os.path.join(DATA_DIR, TARGET_CATEGORY)
    if not os.path.exists(category_folder):
        print(f"Error: Directory '{category_folder}' does not exist.")
        return

    images = [f for f in os.listdir(category_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"No images found in '{category_folder}'.")
        return

    if IMAGE_INDEX >= len(images):
        print(f"Error: Index {IMAGE_INDEX} out of range. Only {len(images)} images available.")
        return

    img_name = images[IMAGE_INDEX]
    img_path = os.path.join(category_folder, img_name)
    print(f"Processing Image: {img_path}")

    # Load image in BGR and convert to RGB
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Keep an explicit uint8 copy for overlay blending
    img_uint8 = img_resized.copy()

    # Normalize image tensor for model input
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_normalized, axis=0)

    # Load Model
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Perform Inference
    prediction_prob = float(model.predict(img_tensor, verbose=0)[0][0])
    label = "TUMOR" if prediction_prob >= 0.5 else "NORMAL"
    confidence = prediction_prob if prediction_prob >= 0.5 else (1 - prediction_prob)

    print(f"Prediction: {label} ({confidence * 100:.2f}%)")

    # Generate Grad-CAM Heatmap
    heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV_LAYER_NAME)

    # Rescale heatmap to match input dimensions
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply Jet color map (Red = High attention, Blue = Low attention)
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap onto uint8 CT scan image (both are now 224x224 uint8 arrays)
    superimposed_img = cv2.addWeighted(img_uint8, 0.6, jet_heatmap, 0.4, 0)

    # Plot & Save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_uint8)
    axes[0].set_title(f"Original CT Scan\n({TARGET_CATEGORY})")
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(superimposed_img)
    axes[2].set_title(f"Overlay Result\nPred: {label} ({confidence*100:.1f}%)")
    axes[2].axis('off')

    plt.tight_layout()
    
    output_image_path = os.path.join(OUTPUT_DIR, f"gradcam_{TARGET_CATEGORY}_{IMAGE_INDEX}.png")
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    print(f"Grad-CAM visualization saved to: {output_image_path}")
    plt.show()


if __name__ == '__main__':
    generate_gradcam()