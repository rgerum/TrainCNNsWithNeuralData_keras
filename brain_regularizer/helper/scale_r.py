from tensorflow import keras
import numpy as np


class ScaleOutput(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.w = self.add_weight(shape=(1,), initializer="ones", trainable=False)

    def call(self, inputs):
        return inputs[..., None] * self.w


class ScaleR(keras.callbacks.Callback):
    def __init__(self, data_gen, model_both, r, r_per_epoch):
        self.data_gen = data_gen
        self.model_both = model_both
        self.r = r
        self.r_per_epoch = r_per_epoch

    def on_train_batch_begin(self, batch, logs=None):
        if self.r_per_epoch is False:
            i, o = self.data_gen[batch]
            loss, class_loss, rsa_loss, class_acc, rsa_acc = self.model_both.evaluate(i, o, verbose=False)
            lambd = self.r * class_loss / rsa_loss
            self.data_gen.lambd = lambd
            self.model.get_layer("rsa_lambda").set_weights([np.ones(1) * lambd])

    def on_epoch_begin(self, epoch, logs=None):
        if self.r_per_epoch is True:
            import numpy as np
            i, o = self.data_gen[np.random.randint(len(self.data_gen))]
            loss, class_loss, rsa_loss, class_acc, rsa_acc = self.model_both.evaluate(i, o, verbose=False)
            lambd = self.r * class_loss / rsa_loss
            self.data_gen.lambd = lambd
