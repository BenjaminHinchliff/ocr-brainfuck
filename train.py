import tensorflow as tf
from tensorflow.keras import layers, regularizers
import datetime
import numpy as np
import matplotlib.pyplot as plt
from model import create_model, IMG_HEIGHT, IMG_WIDTH

data_dir = "./chars/"
batch_size = 32

train_images = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
)

test_images = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
)

model = create_model()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy", "mse"]
)

model.fit(
    train_images,
    epochs=100,
    validation_data=test_images,
    callbacks=[tensorboard_callback],
)

test_loss, test_acc, test_mse = model.evaluate(test_images, verbose=2)

print("\nTest accuracy:", test_acc)

model.save_weights('./weights/weights')
