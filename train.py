import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from dataset import create_data_generators
from model import build_transfer_learning_model

DATA_DIR = 'data/raw/'

def main():
    print("Loading datasets...")
    train_ds, val_ds = create_data_generators(DATA_DIR)

    print("Building model...")
    model = build_transfer_learning_model()
    model.summary()

    epochs = 15
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )

    # Save trained model
    os.makedirs('models', exist_ok=True)
    model.save('models/best_kidney_model.keras')
    print("\nModel saved successfully to 'models/best_kidney_model.keras'.")

    # Evaluate validation predictions
    print("\nEvaluating validation set...")
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_pred_probs) >= 0.5).astype(int)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Tumor']))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Accuracy / Loss Curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()