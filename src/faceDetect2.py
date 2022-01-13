from KerasTrain import KerasTrain
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
from tabulate import tabulate
import cv2

class Detector(object):
    def __init__(self, model=None, imgPath:str = "./images120.png") -> None:
        super().__init__()




        self.model = model

        self.imgPath = imgPath

        self.size = (150, 150)
        self.WINDOW_SIZES = [i for i in range(20, 150, 20)]

        if not self.imgPath.endswith(".png"):
            img = Image.open(self.imgPath)
            print(self.imgPath.split(".")[0] + ".png")
            img.save(self.imgPath.split(".")[0] + ".png")

        # self.imageToPredict = self.loadTensorImg(self.imgPath,self.size)
        self.imageToPredict = cv2.imread(self.imgPath)

            # verify extension



    def loadTensorImg(self, img_path,size):

        img = image.load_img(img_path, target_size=size)
        # (height, width, channels)
        imgTensor = image.img_to_array(img)
        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        imgTensor = np.expand_dims(imgTensor, axis=0)
        # imshow expects values in the range [0, 1]
        imgTensor /= 255.

        return imgTensor





    def get_best_bounding_box(self, img, step=10):
        best_box = None
        best_box_prob = -np.inf
        # loop window sizes: 20x20, 30x30, 40x40...160x160
        for win_size in  self.WINDOW_SIZES:
            for top in range(0, img.shape[0] - win_size + 1, step):
                for left in range(0, img.shape[1] - win_size + 1, step):
                    # compute the (top, left, bottom, right) of the bounding box
                    box = (top, left, top + win_size, left + win_size)

                    # crop the original image
                    cropped_img = img[box[0]:box[2], box[1]:box[3]]

                    # predict how likely this cropped image is dog and if higher
                    # than best save it
                    print('predicting for box %r' % (box, ))
                    print(cropped_img)
                    box_prob = self.model.predictSlice(cropped_img)
                    if box_prob > best_box_prob:
                        best_box = box
                        best_box_prob = box_prob

        return best_box

    def detect(self):
        self.get_best_bounding_box(self.imageToPredict)

if __name__ == "__main__":
    imgPath = "images120.png"
    model = KerasTrain().loadModel()

    detector = Detector(model = model,imgPath=imgPath)
    detector.detect()



