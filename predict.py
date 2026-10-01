import os
import cv2
import numpy as np

# Suppress verbose TensorFlow C++ logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def predict_ct_scan(image_path, model_path='models/best_kidney_model.keras'):
    """
    Loads saved model and classifies an unseen CT scan image as Normal or Tumor.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not read image from path: {image_path}")

    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image file: {image_path}")

    # Preprocess image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, (224, 224))
    normalized_img = np.expand_dims(resized_img, axis=0) / 255.0

    # Predict
    prediction = model.predict(normalized_img, verbose=0)[0][0]

    if prediction >= 0.5:
        confidence = prediction * 100
        result = f"Result: TUMOR detected (Confidence: {confidence:.2f}%)"
    else:
        confidence = (1 - prediction) * 100
        result = f"Result: NORMAL scan (Confidence: {confidence:.2f}%)"

    return result

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # -------------------------------------------------------------
    # 1. Choose folder: 'Normal' or 'Tumor'
    # 2. Choose image index: 0 for 1st image, 1 for 2nd image, etc.
    # -------------------------------------------------------------
    TARGET_CATEGORY = 'Tumor'  # Change to 'Tumor' or 'Normal'
    IMAGE_INDEX = 0             # Change number (0, 1, 2, 3...) to test different images
    
    target_dir = os.path.join(BASE_DIR, 'data', 'raw', TARGET_CATEGORY)
    
    if os.path.exists(target_dir) and os.listdir(target_dir):
        all_images = os.listdir(target_dir)
        
        if IMAGE_INDEX < len(all_images):
            sample_image = os.path.join(target_dir, all_images[IMAGE_INDEX])
            try:
                prediction_result = predict_ct_scan(sample_image)
                print(f"\nTesting File [{TARGET_CATEGORY} #{IMAGE_INDEX + 1}]: {sample_image}")
                print(prediction_result)
            except Exception as e:
                print(f"Notice: {e}")
        else:
            print(f"Index {IMAGE_INDEX} out of range. Folder only has {len(all_images)} images.")
    else:
        print(f"Error: Directory '{target_dir}' does not exist or contains no images.")