from typing import Union
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.optimizers import RMSprop
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
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
from tqdm import trange
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



def save(model, modelPath: str = "model.pkl"):
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
        imgTensor = image.img_to_array(img)
        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        imgTensor = np.expand_dims(imgTensor, axis=0)
        # imshow expects values in the range [0, 1]
        imgTensor /= 255.

        return imgTensor

    def predict(self, img: Union[str, np.ndarray], mode="category", modelPath=None, **kwargs):

        # model = keras.models.load_model("model.h5")
        if modelPath is not None:
            self.model = self.loadModel(modelPath)
        if isinstance(img, str) and not img.endswith(".png"):
            imgPil = Image.open(img)
            imgPil = imgPil.convert("RGB")

            imgPil.save(img.split(".")[0] + ".png")
        # imageToPredict = self.loadTensorImg(imgPath)
        prediction = self.model.predict(img, callbacks=[
            self.tensorboard_callback], workers=self.workers, use_multiprocessing=self.use_multiprocessing, **kwargs)
        # prediction = self.model.predict(imageToPredict, callbacks=[
        #     self.tensorboard_callback], workers=self.workers, use_multiprocessing=self.use_multiprocessing, **kwargs)
        # if mode == "category":
        #     print(decode_predictions(prediction, top=2)[0])
        # else:

        self.displayPredictions(prediction, "", mode)

        return prediction

    def train(self, imageDataGeneratorArgs: dict = {}, modelPath="model.h5", **kwargs):
        aug = ImageDataGenerator(**imageDataGeneratorArgs)

        train_ds = aug.flow_from_directory(
            "converted-images/",
            target_size=self.size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=["Masque", "Pas masque"],
            shuffle=True)
        x_train, y_train = next(iter(train_ds))
        x_test, y_test = next(iter(train_ds))

        self.model: Sequential = Sequential()

        # Input are 150x150 RGB images

        # Input layer of the model
        self.model.add(keras.Input(shape=(150, 150, 3)))
        self.model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
        self.model.add(layers.Conv2D(32, 3, activation="relu"))
        self.model.add(layers.MaxPooling2D(3))

        # Can you guess what the current output shape is at this point? Probably not.
        # Let's just print it:

        # The answer was: (40, 40, 32), so we can keep downsampling...

        self.model.add(layers.Conv2D(32, 3, activation="relu"))
        self.model.add(layers.Conv2D(32, 3, activation="relu"))
        self.model.add(layers.MaxPooling2D(3))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Conv2D(32, 3, activation="relu"))
        self.model.add(layers.Conv2D(32, 3, activation="relu"))
        self.model.add(layers.MaxPooling2D(2))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.GlobalMaxPooling2D())

        # Finally, we add a classification layer.
        self.model.add(layers.Dense(len(self.model_classes_), activation="softmax"))

        # # This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
        # self.model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
        # self.model.add(layers.Conv2D(32, 3, activation="relu"))
        # self.model.add(layers.Conv2D(32, 3, activation="relu"))
        # # a Max pooling operation is for 2D spatial data, it Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input. The window is shifted by strides along each dimension. Calculate the maximum value for each patch of the feature map.
        # self.model.add(layers.MaxPooling2D(3))
        # # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
        # self.model.add(layers.Dropout(0.4))

        # self.model.add(layers.Conv2D(64, 3, activation="relu"))
        # self.model.add(layers.Conv2D(64, 3, activation="relu"))
        # self.model.add(layers.MaxPooling2D(3))
        # self.model.add(layers.Dropout(0.4))

        # self.model.add(layers.Conv2D(128, 3, activation="relu"))
        # self.model.add(layers.Conv2D(128, 3, activation="relu"))
        # self.model.add(layers.MaxPooling2D(2))
        # self.model.add(layers.Dropout(0.4))

        # # The GlobalMaxPooling2D layer produces 1D tensor that comprises of max values for all channels in the images computed along image height and width.
        # self.model.add(layers.GlobalMaxPooling2D())
        # self.model.add(layers.Dense(
        #     len(self.model_classes_), activation=tf.nn.softmax))

        # MODEL VGG16:
        # self.model.add(Conv2D(input_shape=(150, 150, 3), filters=64,
        #                kernel_size=(3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=64, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Conv2D(filters=128, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=128, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Conv2D(filters=256, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=256, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=256, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Conv2D(filters=512, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=512, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=512, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Conv2D(filters=512, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=512, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(Conv2D(filters=512, kernel_size=(
        #     3, 3), padding="same", activation="relu"))
        # self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Flatten())
        # self.model.add(Dense(units=4096,activation="relu"))
        # self.model.add(Dense(units=4096,activation="relu"))
        # self.model.add(Dense(units=len(self.model_classes_), activation="softmax"))

    # MODEL CNN :
        # self.model = Sequential()
        # self.model.add(Conv2D(16, kernel_size = (3, 3),
        # activation = 'relu', input_shape = (150, 150, 3)))
        # self.model.add(Conv2D(32, (3, 3), activation = 'relu'))
        # self.model.add(MaxPooling2D(pool_size = (2, 2)))
        # self.model.add(Dropout(0.2))
        # self.model.add(Flatten())
        # self.model.add(Dense(64, activation = 'relu'))
        # self.model.add(Dropout(0.4))
        # self.model.add(Dense(len(self.model_classes_), activation = 'softmax'))

        # MLP
        # self.model = Sequential()
        # self.model.add(Dense(350, activation='relu'))
        # self.model.add(Dense(50, activation='relu'))
        # self.model.add(Dense(len(self.model_classes_), activation='softmax'))

        # # Configure the model and start training
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2)

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(
                               learning_rate=1e-4),
                           metrics=['accuracy'])
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=self.epochs,
            workers=self.workers,
            use_multiprocessing=self.use_multiprocessing,
            callbacks=[self.tensorboard_callback]
        )
        self.model.save(modelPath)

        plt.plot(history.history['accuracy'], label="train_acc")
        plt.plot(history.history['val_accuracy'], label="test_acc")
        plt.plot(history.history['loss'], label="train_loss")
        plt.plot(history.history['val_loss'], label="test_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig("loss-accuracy.png")

        return history, self.model

    @staticmethod
    def loadModel(path="model.h5"):
        model = keras.models.load_model(path)
        print(model)

        return KerasTrain(
            model=model
        )

    def displayPredictions(self, predictions: np.ndarray, imgPath: str, mode: str):
        print(f"Prediction for :{imgPath}")
        dic = {}
        for i in range(len(self.model_classes_)):
            dic[list(self.model_classes_.keys())[i]] = predictions[0][i]*100

        print(tabulate({k: f"{v:.2f}%" for k, v in dic.items()
                        }.items(), headers=["Class", "Confidence"]))

        idx = np.argmax(predictions, axis=1)[0]

        print(
            f">>> Prediction final \"{list(self.model_classes_.keys())[idx]}\" with {predictions[0][idx]*100:.2f}% confidence\n")

        # values = [x*100 for x in predictions[0]]
        # explode = [0.1 if i == idx else 0 for i in range(len(values))]
        # plt.title(f"Prediction for : {imgPath}")
        # plt.pie(list(dic.values()), explode=explode,labels=list(dic.keys()),autopct='%1.1f%%',shadow=True, startangle=90)
        # plt.axis('equal')

        # plt.savefig(f"pie_chart{idx}.png")
        # plt.clf()

    def predictDirectory(self, dirPath: str = "converted-images"):
        files = []

        dirlist = [dirPath]

        while len(dirlist) > 0:
            for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
                dirlist.extend(dirnames)
                files.extend(map(lambda n: os.path.join(
                    *n), zip([dirpath] * len(filenames), filenames)))

        for i, f in enumerate(files):
            # predictions = self.predict(f)
            split_f = f.split("/")[-1].split(".")

            output_dir = f"{os.path.dirname(f)}/output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            f_output = f"{output_dir}/{split_f[0]}-{i}.{split_f[1]}"

            self.detect_face_and_predict(f, f_output)
            print(f"{f_output} processed")
            # correct_predictions = np.nonzero(predictions == y_test)[0]
            # incorrect_predictions = np.nonzero(predictions != y_test)[0]
            # print(len(correct_predictions)," classified correctly")
            # print(len(incorrect_predictions)," classified incorrectly")

    def detect_face_and_predict(self, img_path: str, output_path: str):
        print("[INFO] loading face detector model...")
        prototxtPath = "deploy.prototxt"
        weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        print("[INFO] loading face mask detector model...")
        # model = KerasTrain().loadModel("model-100-epochs.h5")

        image = cv2.imread(img_path)

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        print("[INFO] computing face detections...")
        net.setInput(blob)
        detections = net.forward()

        for i in trange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (150, 150))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                (mask, withoutMask) = self.model.predict(face)[0]

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)

                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 1)

        cv2.imwrite(output_path, image)
        cv2.waitKey(0)

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
    imgPathMasque = "converted-images/Masque chirurgical/161.png"
    imgPathMasqueTissu = "converted-images/Masque en tissu/44.png"
    imgPathMasqueFFP2 = "converted-images/Masque FFP2/772.png"
    imgPathPasMasque = "converted-images/Pas masque/13.png"
    kerasTrain = KerasTrain(epochs=150, batch_size=150,
                            use_multiprocessing=True, workers=8)
    # kerasTrain.testBatchSize()

    # print(model.model.__dict__)
    # kerasTrain.train(
    #     dict(
    #         rescale=1./255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True
    #     )
    # )
    # kerasTrain.save("model")

    model = KerasTrain().loadModel("model-100-epochs.h5")
    # model.graphPredictDirectory()
    # model = loadPickle()

    # print(model.model_classes_)

    # kerasClassLoad = loadPickle("model.pkl")
    # print(kerasClassLoad.__classes)

    model.predict(imgPathMasque)
    model.predict(imgPathMasqueTissu)
    model.predict(imgPathMasqueFFP2)
    model.predict(imgPathPasMasque)
    # graph_keras_train()
