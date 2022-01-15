from typing import Union
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from PIL import Image
from keras.preprocessing import image
import numpy as np
import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.layers import *
import os
from tqdm import trange
import keras.models
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

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
        self.lr = 1e-5
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        prototxtPath = "deploy.prototxt"
        weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"

        self.net = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1)

        for i, _dir in enumerate(os.listdir("dataset"), 0):
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

        return prediction

    def train(self, imageDataGeneratorArgs: dict = {}, modelPath="model.h5", **kwargs):
        aug = ImageDataGenerator(**imageDataGeneratorArgs)

        files = []
        dirlist = ["dataset/"]

        while len(dirlist) > 0:
            for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
                dirlist.extend(dirnames)
                files.extend(map(lambda n: os.path.join(
                    *n), zip([dirpath] * len(filenames), filenames)))

        data = []
        labels = []
        # loop over the image paths
        for imagePath in files:
            # extract the class label from the filename
            label = imagePath.split(os.path.sep)[-2]

            image = load_img(imagePath, target_size=self.size)
            image = img_to_array(image)
            image = preprocess_input(image)
            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)
        # convert the data and labels to NumPy arrays
        data = np.array(data, dtype="float32")
        labels = np.array(labels)

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)

        (x_train, x_test, y_train, y_test) = train_test_split(data, labels,
                                                              test_size=0.20, stratify=labels, random_state=42)

        baseModel = MobileNetV2(weights="imagenet", include_top=False,
                                input_tensor=Input(shape=(150, 150, 3)))

        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(len(self.model_classes_),
                          activation="softmax", name="output")(headModel)
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        self.model = Model(inputs=baseModel.input, outputs=headModel)
        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False

        opt = Adam(learning_rate=self.lr, decay=self.lr / self.epochs)
        self.model.compile(loss="binary_crossentropy",
                           optimizer=opt,
                           metrics=["accuracy"])
        plot_model(self.model, show_shapes=True)

        history = self.model.fit(
            aug.flow(x_train, y_train, batch_size=self.batch_size),
            validation_data=(x_test, y_test),
            validation_steps=len(x_test) // self.batch_size,
            epochs=self.epochs,
            steps_per_epoch=len(x_train) // self.batch_size,
            workers=self.workers,
            use_multiprocessing=self.use_multiprocessing,
            callbacks=[self.tensorboard_callback]
        )
        self.model.save(modelPath)

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history['accuracy'], label="train_acc")
        plt.plot(history.history['val_accuracy'], label="test_acc")
        plt.plot(history.history['loss'], label="train_loss")
        plt.plot(history.history['val_loss'], label="test_loss")
        plt.xlabel("Epochs #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(f"loss-accuracy-{self.epochs}-{self.batch_size}.png")
        plt.clf()

        return history, self.model

    @staticmethod
    def loadModel(path="model.h5"):
        model = keras.models.load_model(path)
        return KerasTrain(
            model=model
        )

    def displayPredictions(self, predictions: np.ndarray, imgPath: str, coords_msg: str):
        print(f"Prediction for :{imgPath}")
        dic = {}
        for i in range(len(self.model_classes_)):
            dic[list(self.model_classes_.keys())[i]] = predictions[0][i]*100

        print(tabulate({k: f"{v:.2f}%" for k, v in dic.items()
                        }.items(), headers=["Class", "Confidence"]))

        idx = np.argmax(predictions, axis=1)[0]

        print(
            f">>> Prediction final \"{list(self.model_classes_.keys())[idx]}\" with {predictions[0][idx]*100:.2f}% confidence {coords_msg}\n")

    def predictDirectory(self, dirPath: str = "dataset"):
        if os.path.isdir(dirPath):
            files = []

            dirlist = [dirPath]

            while len(dirlist) > 0:
                for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
                    dirlist.extend(dirnames)
                    files.extend(map(lambda n: os.path.join(
                        *n), zip([dirpath] * len(filenames), filenames)))
        else:
            files = [dirPath]

        for i, f in enumerate(files):
            # predictions = self.predict(f)
            if "." not in f:
                continue
            split_f = f.split("/")[-1].split(".")

            output_dir = f"output"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            f_output = f"{output_dir}/{os.path.basename(split_f[0])}-{i}.{split_f[1]}"
            if f_output[-4:] == ".png" or f_output[-4:] == ".jpg":
                self.detect_face_and_predict(f, f_output)
                print(f"{f_output} processed")

    def detect_face_and_predict(self, img_path: str, output_path: str):
        print("[INFO] Loading face detector model...")



        print("[INFO] loading face mask detector model...")

        image = cv2.imread(img_path)
        image = cv2.resize(image, (700, 700))

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        print("[INFO] computing face detections...")
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = image[startY:endY, startX:endX]
                if not len(face):
                    break
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, self.size)
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                prediction = self.model.predict(face)
                (mask, withoutMask) = prediction[0]
                # print(mask, withoutMask)

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                coords_msg = f"on face coords -> ({startX}, {startY}) ({endX}, {endY})"
                self.displayPredictions(prediction, img_path, coords_msg)

                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)

                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        cv2.imwrite(output_path, image)

    def testBatchSize(self):
        batches = []
        testBatches = []
        losses = []
        lossesTests = []
        nbBatches = list(range(1, 257, 32))
        for batch in trange(1, 257, 32):
            trainer = KerasTrain(batch_size=batch, epochs=25, workers=self.workers,
                                 use_multiprocessing=self.use_multiprocessing)
            history, model = trainer.train(
                dict(
                    rotation_range=20,
                    zoom_range=0.15,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.15,
                    horizontal_flip=True,
                    fill_mode="nearest"
                ),
                modelPath=f"model-{batch}.h5"
            )
            batches.append(max(history.history["accuracy"]) * 100)
            testBatches.append(max(history.history["val_accuracy"]) * 100)
            losses.append(max(history.history['loss']))
            lossesTests.append(max(history.history['val_loss']))

        plt.style.use("ggplot")

        plt.plot(nbBatches, batches, label="train_acc")
        plt.plot(nbBatches, testBatches, label="train_test")
        plt.plot(nbBatches, losses, label="train_loss")
        plt.plot(nbBatches, lossesTests, label="test_loss")
        plt.title(f'Model loss/accuracy batch size variation')
        plt.ylabel('Loss/Accuracy')
        plt.xlabel('Batch size #')
        plt.legend(loc='lower left')
        plt.savefig(f"loss_accuracy-batch_size.png")
        plt.clf()


if __name__ == "__main__":
    trainer = KerasTrain()
    trainer.testBatchSize()
