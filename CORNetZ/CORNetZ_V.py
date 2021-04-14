import tensorflow as tf
from tensorflow.keras import layers


class Normalize(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=.75, bias=1.0)


class CORNetZV(object):
    """Implementation of the AlexNet."""

    def __init__(self, num_classes, dropout=1):
        """Create the graph of the AlexNet model.

        Args:
            num_classes: Number of classes in the dataset.
                        keep_prob: Dropout probability.
        """
        self.dropout = dropout
        self.NUM_CLASSES = num_classes

        """ V1 """
        self.v1 = tf.keras.models.Sequential([
            layers.Conv2D(64, [7, 7], strides=[2, 2], activation="relu", bias_initializer="glorot_uniform"),
            Normalize(),
            layers.MaxPool2D([7, 7], strides=[2, 2]),
        ], name="v1")

        """ V2 """
        self.v2 = tf.keras.models.Sequential([
            layers.Conv2D(128, [3, 3], strides=[1, 1], activation="relu", bias_initializer="glorot_uniform"),
            Normalize(),
            layers.MaxPool2D([3, 3], strides=[2, 2]),
        ], name="v2")

        """ V4 """
        self.v4 = tf.keras.models.Sequential([
            layers.Conv2D(256, [3, 3], strides=[1, 1], activation="relu", bias_initializer="glorot_uniform"),
            Normalize(),
            layers.MaxPool2D([3, 3], strides=[2, 2]),
        ], name="v4")

        """ IT """
        self.it = tf.keras.models.Sequential([
            layers.Conv2D(512, [3, 3], strides=[1, 1], activation="relu", bias_initializer="glorot_uniform"),
            Normalize(),
            layers.MaxPool2D([3, 3], strides=[2, 2]),
        ], name="it")

        """ decoder """
        self.flattened = layers.Flatten()

        self.fc6 = layers.Dense(4096, activation="relu", bias_initializer="glorot_uniform")
        self.dropout6 = layers.Dropout(self.dropout)

        self.fc7 = layers.Dense(4096, activation="relu", bias_initializer="glorot_uniform")
        self.dropout7 = layers.Dropout(self.dropout)

        self.fc8 = layers.Dense(self.NUM_CLASSES, activation="linear", name="class", bias_initializer="glorot_uniform")

    def forward(self, x):
        """Create the network graph."""
        x = self.v1(x)
        x = self.v2(x)
        x = self.v4(x)
        x = self.it(x)

        x = self.flattened(x)

        x = self.fc6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        return x

    def forward_V1(self, x):
        """Create the network graph."""
        x = self.v1(x)
        return x

    def forward_V4(self, x):
        x = self.v1(x)
        x = self.v2(x)
        x = self.v4(x)
        return x

    def forward_IT(self, x):
        x = self.v1(x)
        x = self.v2(x)
        x = self.v4(x)
        x = self.it(x)
        return x


from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.layers import Lambda


class CosineDistance(Lambda):
    def __init__(self, **kwargs):
        super().__init__(lambda x: cosine_similarity(x[0], x[1]), output_shape=lambda x: x[0], **kwargs)
