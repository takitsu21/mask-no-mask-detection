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
import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.layers import *
from keras.utils.vis_utils import plot_model
from tensorflow.python.ops.gen_math_ops import mod
from tqdm import trange

class KerasTrain(object):
    def __init__(self, batch_size=32, epochs=10, workers=8) -> None:
        super().__init__()
        # self.mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.size = (150, 150)
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1)

    def loadTensorImg(self, img_path):

        img = image.load_img(img_path, target_size=self.size)
        # (height, width, channels)
        imgTensor = image.img_to_array(img)
        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        imgTensor = np.expand_dims(imgTensor, axis=0)
        # imshow expects values in the range [0, 1]
        imgTensor /= 255.

        return imgTensor

    def predict(self, imgPath: str, mode="category", modelPath="model.h5", **kwargs):

        model = keras.models.load_model("model.h5")
        imageToPredict = self.loadTensorImg(imgPath)
        prediction = model.predict(imageToPredict, callbacks=[
                                   self.tensorboard_callback], workers=self.workers, use_multiprocessing=True, **kwargs)
        # if mode == "category":
        #     print(decode_predictions(prediction, top=2)[0])
        # else:

        # predict = model(imageToPredict[:1]).numpy()
        # print(tf.nn.softmax(predict).numpy())
        print(prediction)
        return prediction

    def train(self, imageDataGeneratorArgs: dict = {}, modelPath="model.h5", **kwargs):
        data_generator = ImageDataGenerator(**imageDataGeneratorArgs)
        train_ds = data_generator.flow_from_directory(
            "converted-images/",
            target_size=self.size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=["Masque", "Pas masque"],
            shuffle=True)

        val_data = data_generator.flow_from_directory(
            "converted-images/",
            target_size=self.size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=["Masque", "Pas masque"],

            shuffle=True)

        print(train_ds.class_indices)

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

        x_train, y_train = next(iter(train_ds))
        x_test, y_test = next(iter(train_ds))

        model: Sequential = Sequential()
        model.add(keras.Input(shape=(150, 150, 3)))  # 250x250 RGB images
        model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
        model.add(layers.Conv2D(32, 3, activation="relu"))
        model.add(layers.Conv2D(32, 3, activation="relu"))
        model.add(layers.MaxPooling2D(3))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:
        # model.summary()

        # The answer was: (40, 40, 32), so we can keep downsampling...

        model.add(layers.Conv2D(64, 3, activation="relu"))
        model.add(layers.Conv2D(64, 3, activation="relu"))
        model.add(layers.MaxPooling2D(3))
        model.add(layers.Conv2D(128, 3, activation="relu"))
        model.add(layers.Conv2D(128, 3, activation="relu"))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.GlobalMaxPooling2D())
        model.add(layers.Dense(2, activation=tf.nn.softmax))

        # Finally, we add a classification layer.

        # And now?
        # model.summary()

        # Now that we have 4x4 feature maps, time to apply global max pooling.

        # print(model.summary())

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=["accuracy"])
        history = model.fit(x_train,
                            y_train,
                            epochs=self.epochs,
                            workers=self.workers,
                            use_multiprocessing=True,
                            verbose=0,
                            callbacks=[self.tensorboard_callback],
                            #   steps_per_epoch=train_ds.samples/train_ds.batch_size,
                            #   validation_steps=val_data.samples/val_data.batch_size,
                            validation_data=val_data,
                            batch_size=self.batch_size)
        model.save(modelPath)

        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return history, model

    def loadModel(self, path="model.h5"):
        model = keras.models.load_model(path)
        print(model)
        plot_model(model, to_file='model_plot.png',
                   show_shapes=True, show_layer_names=True)

        return model

    def testBatchSize(self):
        batches = []
        testBatches = []
        losses = []
        lossesTests = []
        nbBatches = list(range(1, 257, 32))
        for batch in trange(1, 257, 32):
            trainer = KerasTrain(batch_size=batch, epochs=75)
            history, model = trainer.train(
                dict(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True
                )
            )
            batches.append(max(history.history["accuracy"]) * 100)
            testBatches.append(max(history.history["val_accuracy"]) * 100)
            losses.append(max(history.history['loss']))
            lossesTests.append(max(history.history['val_loss']))

        plt.plot(nbBatches, batches)
        plt.plot(nbBatches, testBatches)
        plt.title('model accuracy 75 epochs')
        plt.ylabel('accuracy')
        plt.xlabel('nb batches')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f"model_accuracy-batch.png")
        plt.clf()
        # summarize history for loss
        plt.plot(nbBatches, losses)
        plt.plot(nbBatches, lossesTests)
        plt.title('model loss 75 epochs')
        plt.ylabel('loss')
        plt.xlabel('nb batches')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f"model_loss-batch.png")
        plt.clf()


def graph_keras_train():
    epochs = [x for x in range(1, 50)]
    batch_size = [2**x for x in range(1, 9)]

    x_labels = [0]
    x_ticks = [0]
    i = 0
    for a in epochs:
        for b in batch_size:
            x_labels.append("(" + str(a) + "," + str(b) + ")")
            i += 1
            x_ticks.append(i)

    i = 1

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
    plt.xlabel("(epochs, batch_size)")
    plt.ylabel("Percentage (%)")
    plt.title("Keras variation")
    plt.grid(True)
    x = []
    y_accuracy = []
    y_loss = []

    nbOfGraph = 0
    for epochs in epochs:
        for batch_size in batch_size:
            print(str(epochs) + " " + str(batch_size))
            print(x_ticks[i])
            print(x_labels[i])

            trainer = KerasTrain(epochs, batch_size)
            loss, accu = trainer.train(
                dict(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True
                )
            )

            # trainer.predict(imgPathMasque)
            # error = predictError(pd.read_csv(
            #     "predicted_logistic_regression.csv"))
            # y_error.append(error)

            y_accuracy.append(accu * 100)
            y_loss.append(loss * 100)

            x.append(x_ticks[i])
            print("Accuracy:", accu)

            if i != 0 and i % 9 == 0:
                plt.scatter(x, y_accuracy)
                plt.scatter(x, y_loss)
                plt.legend(["Accuracy", "Error"])

                plt.savefig(
                    f"variation-error_accuracy-{nbOfGraph}.png")
                y_accuracy = []
                y_loss = []
                x = []
                plt.clf()
                plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
                plt.xlabel("(epochs, batch_size)")
                plt.ylabel("Percentage (%)")
                plt.title("Keras variation")
                plt.grid(True)
                nbOfGraph += 1
            i += 1


if __name__ == "__main__":
    imgPathMasque = "converted-images/Masque/images112-bb-155x2-57-73.png"
    imgPathPasMasque = "converted-images/Pas masque/images242-bb-80x17-88-136.png"
    kerasTrain = KerasTrain(epochs=200)
    kerasTrain.testBatchSize()

    #model = kerasTrain.loadModel("model.h5")


    # kerasTrain.train(
    #     dict(
    #         rescale = 1./255,
    #         shear_range = 0.2,
    #         zoom_range = 0.2,
    #         horizontal_flip = True
    #     )
    # )

    # kerasTrain.predict(imgPathMasque)
    # kerasTrain.predict(imgPathPasMasque)

    # graph_keras_train()
