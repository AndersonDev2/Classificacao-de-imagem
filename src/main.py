import os
import sys
import config
import handle_data
import nn
import tensorflow as tf
import random
import numpy as np
from tensorflow import keras
from keras.api._v2 import keras as kerasAPI

base_dir = os.path.dirname(__file__)


def evaluate():
    labels = handle_data.get_labels()
    datatest = handle_data.main()[1]
    x_test, y_test = handle_data.get_x_and_y(datatest)
    model = nn.create_model(len(labels))
    model.load_weights(nn.weights_file_path)
    model.evaluate(x_test, y_test)
    return


def train():
    np.random.seed(40)
    labels = handle_data.get_labels()
    data, datatest = handle_data.main()
    x, y = handle_data.get_x_and_y(data)
    x_test, y_test = handle_data.get_x_and_y(datatest)
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        nn.weights_file_path, monitor="val_accuracy", save_best_only=True, mode="auto")

    model = nn.create_model(len(labels))
    if os.path.exists(nn.weights_file_path):
        model.load_weights(nn.weights_file_path)
    model.fit(x, y, epochs=99999, callbacks=[
        modelCheckpoint], validation_data=(x_test, y_test), batch_size=None)
