import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from pathlib import Path


class ImageDataGeneratorV(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, txt_file, mode="training", batch_size=50, img_size=227, shuffle=True, buffer_size=1000,
                 length=None, crop=False, center=False, scale=False):
        self.txt_file = txt_file
        files = {"V1": "V1_train.txt", "V4": "V4.txt", "IT": "IT.txt"}
        if txt_file in files:
            self.txt_file = Path(__file__).parent / Path("data/V1") / files[txt_file]

        if not Path(self.txt_file).exists():
            from urllib.request import urlopen
            from io import BytesIO
            import zipfile

            print("downloading monkey v1 dataset...")
            url = urlopen("https://github.com/rgerum/TrainCNNsWithNeuralData_keras/releases/download/v0.0/monkey_v1.zip")
            data = url.read()
            file = BytesIO(data)
            Path("data/V1").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file) as zip_ref:
                zip_ref.extractall("data/V1")

        self.img_size = int(img_size)
        self.shuffle = shuffle
        self.crop = crop
        self.center = center
        self.scale = scale

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.RSAs)
        self.batch_size = batch_size

        self.overhang = 0
        self.random = np.random.default_rng()
        if length is not None:
            self.overhang = length * self.batch_size - self.batch_size
            self.length = length
        else:
            self.length = int(np.floor(self.data_size / self.batch_size))

        self.on_epoch_end()

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img1_paths = []
        self.img2_paths = []
        self.RSAs = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split()
                self.img1_paths.append(Path("data/V1") / items[0])
                self.img2_paths.append(Path("data/V1") / items[1])
                self.RSAs.append(float(items[2]))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.length

    img1 = None
    img2 = None
    rsa = None
    last_index = None
    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.last_index == index:
            return [self.img1, self.img2], self.rsa

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        if self.img1 is None:
            self.img1 = np.array([self.imread(self.img1_paths[k]) for k in indexes])
            self.img2 = np.array([self.imread(self.img2_paths[k]) for k in indexes])
            self.rsa = np.array([self.RSAs[k] for k in indexes], dtype=np.float32)[:, None]
        else:
            for i, k in enumerate(indexes):
                self.img1[i] = self.imread(self.img1_paths[k])
                self.img2[i] = self.imread(self.img2_paths[k])
                self.rsa[i] = self.RSAs[k]
        self.last_index = index

        return [self.img1, self.img2], self.rsa

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.concatenate((self.random.permutation(self.data_size), self.random.permutation(self.data_size)[:self.overhang]))

    def imread(self, filename):
        # load and preprocess the image
        if self.crop is True:
            img_resized = Image.open(filename).convert('RGB').crop((146, 60, 512, 426))
            img_resized = np.array(img_resized.resize([self.img_size, self.img_size], Image.BILINEAR))
            img_resized = img_resized.astype(np.float32)
        else:
            img_resized = keras.preprocessing.image.load_img(filename, color_mode="rgb",
                                                             target_size=(int(self.img_size), int(self.img_size)),
                                                             interpolation="bilinear")
            img_resized = keras.preprocessing.image.img_to_array(img_resized)

        if self.center is True:
            img_resized -= 128
        if self.scale is True:
            img_resized /= 128

        return img_resized
