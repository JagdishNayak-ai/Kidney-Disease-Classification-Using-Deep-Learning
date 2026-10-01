import os
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def create_data_generators(data_dir):
    """
    Loads training and validation datasets using tf.keras.utils.image_dataset_from_directory
    (Compatible with Keras 3 & TensorFlow 2.x).
    """
    # Load Training Split (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Load Validation Split (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Rescale pixel values from [0, 255] to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Optimize data pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds