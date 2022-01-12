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
import os
from tqdm import trange
import types
import tempfile
import keras.models
import pickle


def save(model, modelPath: str="model.pkl"):
    pickle.dump(model, open(modelPath, "wb+"))
    print(f"save {modelPath}")


class KerasTrain(object):
    def __init__(self, model=None, batch_size=32, epochs=10, workers=1, use_multiprocessing=False) -> None:
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.model_classes_ = {}
        self.size = (150, 150)
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1)

        for i, _dir in enumerate(os.listdir("converted-images"), 0):
            self.model_classes_[_dir] = i


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

        # model = keras.models.load_model("model.h5")
        print(not imgPath.endswith(".png"))
        if not imgPath.endswith(".png"):
            img = Image.open(imgPath)
            print(imgPath.split(".")[0] + ".png")
            img.save(imgPath.split(".")[0] + ".png")
        imageToPredict = self.loadTensorImg(imgPath)
        prediction = self.model.predict(imageToPredict, callbacks=[
                                   self.tensorboard_callback], workers=self.workers, use_multiprocessing=self.use_multiprocessing, **kwargs)
        # if mode == "category":
        #     print(decode_predictions(prediction, top=2)[0])
        # else:

        # predict = model(imageToPredict[:1]).numpy()
        # print(tf.nn.softmax(predict).numpy())
        # print(prediction)
        self.displayPredictions(prediction,imgPath)
        # print(self.model.predict_classes(imageToPredict))

        return prediction


    def train(self, imageDataGeneratorArgs: dict = {}, modelPath="model.h5", **kwargs):
        data_generator = ImageDataGenerator(**imageDataGeneratorArgs)
        train_ds = data_generator.flow_from_directory(
            "converted-images/",
            target_size=self.size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        self.model_classes_ = train_ds.class_indices

        val_data = data_generator.flow_from_directory(
            "converted-images/",
            target_size=self.size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)


        x_train, y_train = next(iter(train_ds))
        x_test, y_test = next(iter(train_ds))

        self.model: Sequential = Sequential()
        self.model.add(keras.Input(shape=(150, 150, 3)))  # 250x250 RGB images
        self.model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
        self.model.add(layers.Conv2D(32, 3, activation="relu"))
        self.model.add(layers.Conv2D(32, 3, activation="relu"))
        self.model.add(layers.MaxPooling2D(3))




        self.model.add(layers.Conv2D(64, 3, activation="relu"))
        self.model.add(layers.Conv2D(64, 3, activation="relu"))
        self.model.add(layers.MaxPooling2D(3))
        self.model.add(layers.Conv2D(128, 3, activation="relu"))
        self.model.add(layers.Conv2D(128, 3, activation="relu"))
        self.model.add(layers.MaxPooling2D(2))
        self.model.add(layers.GlobalMaxPooling2D())
        self.model.add(layers.Dense(4, activation=tf.nn.softmax))



        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=["accuracy"])
        history = self.model.fit(x_train,
                            y_train,
                            epochs=self.epochs,
                            workers=self.workers,
                            use_multiprocessing=self.use_multiprocessing,
                            verbose=1,
                            callbacks=[self.tensorboard_callback],
                            #   steps_per_epoch=train_ds.samples/train_ds.batch_size,
                            #   validation_steps=val_data.samples/val_data.batch_size,
                            validation_data=val_data,
                            batch_size=self.batch_size)

        self.model.save(modelPath)


        return history, self.model

    @staticmethod
    def loadModel(path="model.h5"):
        model = keras.models.load_model(path)
        print(model)
        # plot_model(model, to_file='model_plot.png',
        #            show_shapes=True, show_layer_names=True)

        return KerasTrain(
            model=model
        )

    def displayPredictions(self, predictions: np.ndarray, imgPath:str):
        print(f"Prediction for :{imgPath}")
        classes = self.model_classes_
        # print(classes)

        # print(list(classesPred))
        for i in range(len(classes)):
            # percentPred = predictions[i][np.argmax(predictions, axis=1)][0] * 100
            print(f"Prediction : {list(classes.keys())[i]} | Confidence : {predictions[0][i]*100:.2f}%")

        # print(predictions)
        idx = np.argmax(predictions, axis=1)[0]

        print(f">>> Prediction final \"{list(classes.keys())[idx]}\" with {predictions[0][idx]*100:.2f}% confidence\n")



    def graphPredictDirectory(self, dirPath: str="converted-images"):
        files = []

        dirlist = [dirPath]

        while len(dirlist) > 0:
            for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
                dirlist.extend(dirnames)
                files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames)))


        for f in files :
            predictions = self.predict(f)
            # correct_predictions = np.nonzero(predictions == y_test)[0]
            # incorrect_predictions = np.nonzero(predictions != y_test)[0]
            # print(len(correct_predictions)," classified correctly")
            # print(len(incorrect_predictions)," classified incorrectly")


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


def loadPickle(path: str="model.pkl"):
    model = pickle.load(open(path), "rb", encoding="utf-8")
    return model

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
    imgPathMasque = "converted-images/Masque/images307-bb-48x7-141-155.png"
    imgPathPasMasque = "converted-images/Pas masque/images159-bb-209x46-75-60.png"
    imgPathMasque2 = "converted-images/Masque/images124-bb-225x94-40-54.png"
    telechargement = "img/tele.png"
    kerasTrain = KerasTrain(epochs=25, batch_size=50, use_multiprocessing=True, workers=8)
    # kerasTrain.testBatchSize()



    # print(model.model.__dict__)
    kerasTrain.train(
        dict(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    )
    # kerasTrain.save("model")

    model = KerasTrain().loadModel()
    # model.graphPredictDirectory()
    # model = loadPickle()

    # print(model.model_classes_)

    # kerasClassLoad = loadPickle("model.pkl")
    # print(kerasClassLoad.__classes)

    model.predict(imgPathMasque)
    model.predict(imgPathPasMasque)
    model.predict(imgPathMasque2)
    model.predict(telechargement)
    # graph_keras_train()
