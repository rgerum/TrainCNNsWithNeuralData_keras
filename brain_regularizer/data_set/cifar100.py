import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import pandas as pd

# from tensorflow.contrib.data import Dataset
from tensorflow.python.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

CIF100_MEAN = np.array([129.3, 124.1, 112.4], dtype=np.float32)

def Cifar100(img_size, batch_size, scale=True):
    num_classes = 100
    tr_data = ImageDataGenerator("train.txt",
                     img_size=img_size, batch_size=batch_size, num_classes=num_classes, shuffle=True, scale=scale)
    val_data = ImageDataGenerator("val.txt",
                     img_size=img_size, batch_size=batch_size, num_classes=num_classes, shuffle=False, scale=scale)
    return tr_data, val_data

class ImageDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, txt_file, batch_size, num_classes, img_size=227, shuffle=True, scale=False):
        """Create a new ImageDataGenerator.

        Receives a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensorFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Whether or not to shuffle the data in the dataset and the
                initial file list.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = Path(__file__).parent / "tmp_data" / "CIFAR100" / txt_file

        print(Path(self.txt_file), Path(self.txt_file).exists())
        if not Path(self.txt_file).exists():
            from urllib.request import urlopen
            from io import BytesIO
            import zipfile

            print("downloading cifar100 dataset...")
            url = urlopen("https://github.com/rgerum/TrainCNNsWithNeuralData_keras/releases/download/v0.0/cifar100.zip")
            data = url.read()
            file = BytesIO(data)
            self.txt_file.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file) as zip_ref:
                zip_ref.extractall(self.txt_file.parent)

        self.num_classes = num_classes
        self.img_size = img_size
        self.shuffle = shuffle
        self.scale = scale

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        self.batch_size = batch_size

        self.on_epoch_end()

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(Path("data/CIFAR100") / items[0])
                self.labels.append(int(items[1]))
        self.labels = np.array(self.labels)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.data_size / self.batch_size))

    img = None
    img_labels = None
    last_index = None
    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.last_index == index:
            return self.img, self.img_labels

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        if self.img is None:
            self.img = np.array([self.imread(self.img_paths[k]) for k in indexes])
            self.img_labels = keras.utils.to_categorical(self.labels[indexes], num_classes=self.num_classes)
        else:
            for i, k in enumerate(indexes):
                self.img[i] = self.imread(self.img_paths[k])
            self.img_labels[:] = keras.utils.to_categorical(self.labels[indexes], num_classes=self.num_classes)
        self.last_index = index

        return self.img, self.img_labels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.data_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def imread(self, filename):
        # load and preprocess the image
        img_resized = keras.preprocessing.image.load_img(filename, color_mode="rgb",
                                                         target_size=(int(self.img_size), int(self.img_size)),
                                                         interpolation="bilinear")
        img_resized = keras.preprocessing.image.img_to_array(img_resized)
        img_resized -= CIF100_MEAN
        if self.scale:
            img_resized /= 128

        # RGB -> BGR
        img_resized = img_resized[:, :, ::-1]

        return img_resized