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
import argparse
from src.KerasTrain import KerasTrain



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict classes of an image')
    parser.add_argument('-i', '--image', type=str,
                        help="Image path", dest="img_path", default=None)
    parser.add_argument('-d', '--description', type=str,
                        help="Description of the movie", dest="description", default=None)
    parser.add_argument('-tr', '--train', type=bool,
                        help="Train the model", dest="train", default=False)
    parser.add_argument('-mp', '--model_path', type=str, help="Path to the model",
                        dest="model_path", default="model.h5")

    parser.add_argument('-e', '--epochs', type=int,
                        help="Epoch size (recommended size for more accuracy: 150)", dest="epochs", default=30)
    parser.add_argument('-b', '--batch_size', type=int,
                        help="Batch size", dest="batch_size", default=75)
    parser.add_argument("-w", "--workers", type=int,
                        help="Number of workers", dest="workers", default=1)
    parser.add_argument("-dir", "--dir_predict_path", type=str,
                        help="Path to the directory with images", dest="dir_predict_path", default=None)
    args = parser.parse_args()
    if args.train:
        kerasTrain = KerasTrain(epochs=args.epochs, batch_size=args.batch_size,
                                use_multiprocessing=True if args.workers > 1 else False, workers=args.workers)
        kerasTrain.train(
            dict(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest"
            ),
            modelPath=args.model_path
        )
    # kerasTrain.save("model")
    if args.img_path is not None:
        model = KerasTrain().loadModel(path=args.model_path)
        model.detect_face_and_predict(args.img_path, f"output-{args.img_path}")
        # model.predict(args.img_path)


    if args.dir_predict_path is not None:
        model = KerasTrain().loadModel(path=args.model_path)
        model.predictDirectory(dirPath=args.dir_predict_path)

    # graph_keras_train()
