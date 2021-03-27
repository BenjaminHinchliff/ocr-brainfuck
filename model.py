import tensorflow as tf
from tensorflow.keras import layers

NUM_CLASSES = 9
IMG_WIDTH = 28
IMG_HEIGHT = 28

def create_model():
    return tf.keras.Sequential(
        [
            layers.experimental.preprocessing.Rescaling(1.0 / 255),
            layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT)),
            layers.Dense(units=300, activation="elu"),
            layers.Dropout(0.5),
            layers.Dense(units=300, activation="elu"),
            layers.Dropout(0.5),
            layers.Dense(units=100, activation="elu"),
            layers.Dense(units=NUM_CLASSES),
        ]
    )
