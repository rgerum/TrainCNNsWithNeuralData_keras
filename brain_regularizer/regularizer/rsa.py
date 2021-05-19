import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.layers import Lambda
import numpy as np
from brain_regularizer.helper import constructModelCopy, MergedGenerators
import itertools
from brain_regularizer.helper import ScaleOutput


def np_cosine_similarity(x, y):
    return np.sum(x*y, axis=1)/(np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1))


class CosineDistance(Lambda):
    def __init__(self, **kwargs):
        super().__init__(lambda x: cosine_similarity(x[0], x[1]), output_shape=lambda x: x[0], **kwargs)


class GeneratorRSA(keras.utils.Sequence):
    def __init__(self, generator, batch_size):
        self.generator = generator
        self.batch_size = batch_size
        self.pairs = list(itertools.combinations(np.arange(0, len(self.generator)), 2))
        self.random = np.random.default_rng()
        self.on_epoch_end()

        img1, neurons1 = self.generator[0]
        self.images1 = np.zeros((self.batch_size, img1.shape[1], img1.shape[2], img1.shape[3]))
        self.images2 = np.zeros((self.batch_size, img1.shape[1], img1.shape[2], img1.shape[3]))
        self.neuro1 = np.zeros((self.batch_size, neurons1.shape[1]))
        self.neuro2 = np.zeros((self.batch_size, neurons1.shape[1]))

    def __len__(self):
        return len(self.pairs)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self.random.permutation(len(self.pairs))
        #self.indexes = np.concatenate((self.random.permutation(self.data_size), self.random.permutation(self.data_size)[:self.overhang]))

    last_index = None
    def __getitem__(self, index):
        if index == self.last_index:
            return [self.images1, self.images2], self.rsas
        for i, item in enumerate(self.indexes[index * self.batch_size:(index + 1) * self.batch_size]):
            self.images1[i], self.neuro1[i] = self.generator[self.pairs[item][0]]
            self.images2[i], self.neuro2[i] = self.generator[self.pairs[item][1]]
        self.rsas = cosine_similarity(self.neuro1, self.neuro2)
        self.last_index = index
        return [self.images1, self.images2], self.rsas



def RSA(model, layer, tr_data, val_data, brain_data):
    # take two images as input
    img1 = tf.keras.layers.Input(model.input.shape[1:], name="img1")
    img2 = tf.keras.layers.Input(model.input.shape[1:], name="img2")

    # process then with the V1 part of the model
    submodel = constructModelCopy(model, layer)
    img1_v1 = submodel(img1)
    img2_v2 = submodel(img2)

    # calculate the cosine distance
    rsa = CosineDistance(name="rsa")([tf.keras.layers.Flatten()(img1_v1), tf.keras.layers.Flatten()(img2_v2)])
    # multiply by weighting factor
    lambd_rsa = ScaleOutput(name="rsa_lambda")(rsa)

    brain_data = GeneratorRSA(brain_data, batch_size=tr_data.batch_size)
    joined_train_data = MergedGenerators(tr_data, brain_data, use_min_length=True)
    joined_validation_data = MergedGenerators(val_data, brain_data, use_min_length=True)

    model_both = keras.models.Model(inputs=[model.input, img1, img2], outputs=[model.output, lambd_rsa])
    print(model_both.summary())
    return model_both, joined_train_data, joined_validation_data

