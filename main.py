import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def train():
    data_generator = ImageDataGenerator()
    train_data = data_generator.flow_from_directory(
        "converted-images/",
        target_size=(150, 150),
        batch_size = 32,
        class_mode = 'categorical',
        shuffle = True)

    val_data = data_generator.flow_from_directory(
        "converted-images/",
        target_size=(150, 150),
        batch_size=32,
        class_mode = 'categorical',
        shuffle = True)

    model = tf.keras.models.Sequential([
        Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                             metrics=['accuracy'])

    model.fit(train_data,
          steps_per_epoch=train_data.samples/32,
          validation_data=val_data,
          validation_steps=val_data.samples/32,
          batch_size=32,
          epochs=5,
          )

if __name__ == "__main__":
    train()