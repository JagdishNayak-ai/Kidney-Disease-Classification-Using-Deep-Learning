import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_kidney_model.keras')

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Save architecture summary to a text file with UTF-8 encoding
output_file = os.path.join(BASE_DIR, 'model_architecture.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print(f"Model architecture saved successfully to '{output_file}'!")