import os
import config
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras as kerasAPI


base_dir = os.path.dirname(__file__)
weights_file_path = os.path.join(base_dir, "model/weights.hdf5")


def create_model(len_output):
    model = kerasAPI.Sequential(
        [kerasAPI.Input(shape=(config.size, config.size, 3))])
    for unit in [16, 32, 64, 192]:
        model.add(kerasAPI.layers.Conv2D(unit, (3, 3), padding="same",
                  activation=kerasAPI.activations.relu))
        model.add(kerasAPI.layers.MaxPooling2D())
        model.add(kerasAPI.layers.Dropout(0.2))
    model.add(kerasAPI.layers.Flatten())
    for unit in [512, 256, 128, 64, 32]:
        model.add(kerasAPI.layers.Dense(
            unit, activation=kerasAPI.activations.relu))
        model.add(kerasAPI.layers.Dropout(0.2))
    model.add(kerasAPI.layers.Dense(
        len_output, activation=kerasAPI.activations.softmax))
    model.compile(optimizer=kerasAPI.optimizers.Adam(),
                  metrics=["accuracy"],
                  loss=kerasAPI.losses.SparseCategoricalCrossentropy())
    return model
