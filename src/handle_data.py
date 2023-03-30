import os
import sys
import numpy as np
import tensorflow as tf
import config
import random
import cv2
from PIL import Image

base_dir = os.path.dirname(__file__)


def show_image(image, title="window"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(path_file):
    img = np.array(Image.open(path_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.img_shape)
    img = img/255.0
    return img


def get_labels():
    labels = os.listdir(os.path.join(base_dir, "dataset/train"))
    return labels


def get_x_and_y(data):
    random.shuffle(data)
    x, y = [], []
    for i in range(len(data)):
        x1 = data[i]["image"]
        y1 = data[i]["label"]

        x.append(x1)
        y.append(y1)
    return np.array(x), np.array(y)


def main(len1=0):
    dataset_train = os.path.join(base_dir, "dataset/train")
    dataset_test = os.path.join(base_dir, "dataset/test")
    if len1 == 0:
        len_for_each_label = -1
    else:
        len_for_each_label = int(len1/len(os.listdir(dataset_train)))
    data = []
    # train data
    for i, folder in enumerate(os.listdir(dataset_train)):
        sFolder = os.path.join(dataset_train, folder)
        for file in os.listdir(sFolder)[:len_for_each_label]:
            img = load_image(os.path.join(sFolder, file))
            sData = {
                "image": img,
                "label": i,
                "folder_name": folder
            }
            data.append(sData)
    # test data
    datatest = []
    for i, folder in enumerate(os.listdir(dataset_test)):
        sFolder = os.path.join(dataset_test, folder)
        for file in os.listdir(sFolder)[:len_for_each_label]:
            img = load_image(os.path.join(sFolder, file))
            sData = {
                "image": img,
                "label": i,
                "folder_name": folder
            }
            datatest.append(sData)
    return data, datatest
