import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
from keras.preprocessing import image
import numpy as np


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(256, 256))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor /= 255.

    return img_tensor


class KerasTrain(object):
    def __init__(self) -> None:
        super().__init__()

    def dataAugmentation(self):
        data_augmentation: Sequential = Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
            ]
        )

    def train(self, imgPath, modelPath="model.h5"):
        data_generator = ImageDataGenerator()
        train_ds = data_generator.flow_from_directory(
            "converted-images/",
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical',
            classes=["Masque", "Pas masque"],
            shuffle=True)
        print(train_ds.classes)

        # validation_generator = data_generator.flow_from_directory(
        #     "/converted-images",
        #     target_size=(150, 150),
        #     batch_size=32,
        #     class_mode = 'categorical',
        #     shuffle = True)

        # train_ds = utils.image_dataset_from_directory(
        #     directory='converted-images/',
        #     labels='inferred',
        #     label_mode='categorical',
        #     batch_size=10,
        #     image_size=(256, 256))
        import datetime
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        x_train, y_train = next(iter(train_ds))
        x_test, y_test = next(iter(train_ds))

        model: Sequential = keras.Sequential()
        model.add(keras.Input(shape=(256, 256, 3)))  # 250x250 RGB images
        model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
        model.add(layers.Conv2D(32, 3, activation="relu"))
        model.add(layers.MaxPooling2D(3))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        model.summary()

        # The answer was: (40, 40, 32), so we can keep downsampling...

        model.add(layers.Conv2D(32, 3, activation="relu"))
        model.add(layers.Conv2D(32, 3, activation="relu"))
        model.add(layers.MaxPooling2D(3))
        model.add(layers.Conv2D(32, 3, activation="relu"))
        model.add(layers.Conv2D(32, 3, activation="relu"))
        model.add(layers.MaxPooling2D(2))

        # And now?
        model.summary()

        # Now that we have 4x4 feature maps, time to apply global max pooling.
        model.add(layers.GlobalMaxPooling2D())

        # Finally, we add a classification layer.
        model.add(layers.Dense(2))

        # x = tf.ones((1, 4))
        # y = model(x)
        # print("Number of weights after calling the model:", len(model.weights))  # 6

        print(model.summary())

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), workers=8,
                  use_multiprocessing=True, verbose=1, callbacks=[tensorboard_callback])
        model.save("model.h5")
        imageToPredict = load_image(imgPath)

        prediction = model.predict(imageToPredict, callbacks=[tensorboard_callback])
        print(prediction)
        # print(decode_predictions(prediction, top=3)[0])


if __name__ == "__main__":
    imgPath = "converted-images/Masque/images107-bb-29x35-27-36.png"
    kerasTrain = KerasTrain()
    kerasTrain.train(imgPath)
