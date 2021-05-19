import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from pathlib import Path


NUMBER_IMAGES = 956

class MonkeyV1(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, mode="training", batch_size=50, img_size=227, shuffle=True, buffer_size=1000,
                 length=None, crop=False, center=False, scale=False):

        self.data_folder = Path(__file__).parent / "tmp_data" / "V1"
        self.neuron_data_file = self.data_folder / "v1_processed" / "neural_data_pca.npy"
        self.image_folder = self.data_folder / "naturalimages_227_227_3"

        print(Path(self.image_folder), Path(self.image_folder).exists())
        if not Path(self.image_folder).exists():
            from urllib.request import urlopen
            from io import BytesIO
            import zipfile

            print("downloading monkey v1 dataset...")
            url = urlopen("https://github.com/rgerum/TrainCNNsWithNeuralData_keras/releases/download/v0.0/monkey_v1.zip")
            data = url.read()
            file = BytesIO(data)
            Path(self.data_folder).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file) as zip_ref:
                zip_ref.extractall(self.data_folder)

        self.neuron_data = np.load(self.neuron_data_file)
        self.neuron_data = self.neuron_data.transpose(1, 0, 2)
        self.neuron_data = self.neuron_data.reshape(self.neuron_data.shape[0], -1)

        self.img_size = int(img_size)
        self.shuffle = shuffle
        self.crop = crop
        self.center = center
        self.scale = scale

        # number of samples in the dataset
        self.data_size = self.neuron_data.shape[0]
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

    img = None
    img2 = None
    rsa = None
    last_index = None
    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.last_index == index:
            return self.img, self.rsa

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.img = np.array([self.imread(self.image_folder / f"{k}.png") for k in indexes])
        self.rsa = np.array([self.neuron_data[k] for k in indexes], dtype=np.float32)
        self.last_index = index

        return self.img, self.rsa

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self.random.permutation(self.data_size)

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
