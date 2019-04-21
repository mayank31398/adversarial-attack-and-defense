import os
from datetime import datetime

import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

from params import CLASSES

plt.switch_backend("QT5Agg")


class DEVNAGRIDataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        folders = os.listdir(os.path.join("Data", "Devnagri", "Train"))
        for i in tqdm(range(len(folders))):
            folder = folders[i]

            # Train
            files = os.listdir(os.path.join(
                "Data", "Devnagri", "Train", folder))
            for file in files:
                image = cv2.imread(os.path.join(
                    "Data", "Devnagri", "Train", folder, file))

                image = image[2:30, 2:30, ...]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                self.x_train.append(image)
                self.y_train.append(i)

            # Test
            files = os.listdir(os.path.join(
                "Data", "Devnagri", "Test", folder))
            for file in files:
                image = cv2.imread(os.path.join(
                    "Data", "Devnagri", "Test", folder, file))

                image = image[2:30, 2:30, ...]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                self.x_test.append(image)
                self.y_test.append(i)

        self.x_train = np.array(self.x_train)
        self.x_train = self.x_train / 255
        self.train_batches = self.x_train.shape[0] // batch_size

        self.y_train = np.array(self.y_train)

        self.x_test = np.array(self.x_test)
        self.x_test = self.x_test / 255
        self.test_batches = self.x_test.shape[0] // batch_size

        self.y_test = np.array(self.y_test)

    def GetTrainLoader(self):
        count = 0
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)

        while(True):
            if(count < self.train_batches):
                mini_batch_x = self.x_train[count *
                                            self.batch_size: (count + 1) * self.batch_size, ...]
                mini_batch_x = np.expand_dims(mini_batch_x, axis=3)

                mini_batch_y = self.y_train[count *
                                            self.batch_size: (count + 1) * self.batch_size, ...]

                count += 1
                yield mini_batch_x, mini_batch_y
            else:
                self.x_train, self.y_train = shuffle(
                    self.x_train, self.y_train)
                count = 0

    def GetTestLoader(self):
        count = 0
        self.x_test, self.y_test = shuffle(self.x_test, self.y_test)

        while(True):
            if(count < self.test_batches):
                mini_batch_x = self.x_test[count *
                                           self.batch_size: (count + 1) * self.batch_size, ...]
                mini_batch_x = np.expand_dims(mini_batch_x, axis=3)

                mini_batch_y = self.y_test[count *
                                           self.batch_size: (count + 1) * self.batch_size, ...]

                count += 1
                yield mini_batch_x, mini_batch_y
            else:
                self.x_test, self.y_test = shuffle(self.x_test, self.y_test)
                count = 0


def MakeOneHot(x: np.ndarray):
    y = np.zeros((x.shape[0], CLASSES))
    y[np.arange(x.shape[0]), x] = 1
    return y


def MakeNotSoHot(x: np.ndarray):
    return x.argmax(axis=1)


def DisplayTrain(iteration, loss, accuracy):
    print("Train Iteration: %d, Loss: %f, Accuracy: %f\n\n" % (
        iteration,
        loss,
        accuracy
    ))


def DisplayTest(loss, accuracy):
    print("Test Loss: %f, Accuracy: %f\n\n" % (
        loss,
        accuracy
    ))


def CodeLogger(files):
    full_code = ""
    for file in files:
        with open(file, "r") as file_io:
            full_code += "####################$$$$$$$$$$$$$$$$$$$$ " + file + " $$$$$$$$$$$$$$$$$$$$####################\n"
            code = file_io.readlines()
            for line in code:
                if("api_key" not in line):
                    full_code += line
        if(file != files[-1]):
            full_code += "\n\n"
    
    return full_code