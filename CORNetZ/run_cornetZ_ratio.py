import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

from CORNetZ_V import CORNetZV, CosineDistance
from datagenerator import ImageDataGenerator, MergedGenerators, TrainingHistory, getOutputPath
from datagenerator_v import ImageDataGeneratorV


def run_net(args):
    print(args)
    np.random.seed(1234)

    # data
    tr_data = ImageDataGenerator("train.txt",
                     img_size=args.img_size, batch_size=args.batch_size, num_classes=args.num_classes, shuffle=True)
    val_data = ImageDataGenerator("val.txt",
                     img_size=args.img_size, batch_size=args.batch_size, num_classes=args.num_classes, shuffle=False)

    # the input image from the
    img = tf.keras.layers.Input([args.img_size, args.img_size, 3], name="img")

    # Initialize model
    cornet = CORNetZV(args.num_classes, args.dropout_rate)

    cls = cornet.forward(img)

    model_class = keras.models.Model(inputs=img, outputs=cls)
    opti = keras.optimizers.SGD(learning_rate=args.learning_rate)

    def logit_cost(y, x):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))
    model_class.compile(optimizer=opti, loss=logit_cost, metrics=["accuracy"])
    print(model_class.summary())

    # generate the output path
    output_path = getOutputPath(args)

    # callbacks
    if args.no_log is False:
        calbacks = [
            keras.callbacks.ModelCheckpoint(str(output_path / "weights.hdf5"), save_best_only=True, save_weights_only=True),
            TrainingHistory(output_path),
        ]
    else:
        callbacks = []

    if args.v != 'None':
        # get the data generator for the images from the neuronal data
        v_data = ImageDataGeneratorV(args.v, batch_size=args.v_batch_size, length=len(tr_data), crop=args.crop)

        # take two images as input
        img1 = tf.keras.layers.Input([args.img_size, args.img_size, 3], name="img1")
        img2 = tf.keras.layers.Input([args.img_size, args.img_size, 3], name="img2")

        # process then with the V1 part of the model
        img1_v1 = cornet.forward_V1(img1)
        img2_v2 = cornet.forward_V1(img2)

        # calculate the cosine distance
        rsa = CosineDistance(name="rsa")([tf.keras.layers.Flatten()(img1_v1), tf.keras.layers.Flatten()(img2_v2)])
        # multiply by weighting factor
        lambd = tf.keras.layers.Input([1], name="lambd")
        lambd_rsa = tf.keras.layers.Lambda(lambda x: x[0]*x[1], name="rsa_lambda")([rsa, lambd])

        model_both = keras.models.Model(inputs=[img, img1, img2, lambd], outputs=[cls, lambd_rsa])
        print(model_both.summary())
        #tf.keras.utils.plot_model(model_both, to_file='model.png', show_shapes=True, expand_nested=True)

        model_both.compile(optimizer=opti,
                           loss={"class": logit_cost, "rsa_lambda": "mean_absolute_error"},
                           metrics=["accuracy"])
        print(model_both.summary())

        model_rsa = keras.models.Model(inputs=[img1, img2], outputs=[rsa])
        model_rsa.compile(optimizer="adam", loss={"rsa": "mean_absolute_error"}, metrics=["accuracy"])

        class CustomCallback(keras.callbacks.Callback):
            def __init__(self, data_gen):
                self.data_gen = data_gen

            def on_train_batch_begin(self, batch, logs=None):
                if args.r_per_epoch is False:
                    i, o = self.data_gen[batch]
                    loss, class_loss, rsa_loss, class_acc, rsa_acc = model_both.evaluate(i, o, verbose=False)
                    lambd = args.r * class_loss / rsa_loss
                    self.data_gen.lambd = lambd

            def on_epoch_begin(self, epoch, logs=None):
                if args.r_per_epoch is True:
                    import numpy as np
                    i, o = self.data_gen[np.random.randint(len(self.data_gen))]
                    loss, class_loss, rsa_loss, class_acc, rsa_acc = model_both.evaluate(i, o, verbose=False)
                    lambd = args.r * class_loss / rsa_loss
                    self.data_gen.lambd = lambd

        training_input = MergedGenerators(tr_data, v_data)
        training_input2 = MergedGenerators(val_data, v_data, use_min_length=True)

        # the initial epochs with teacher
        model_both.fit(training_input, validation_data=training_input2, shuffle=False, epochs=10, callbacks=[CustomCallback(training_input)]+calbacks)
        # the rest of the epochs without a teacher
        model_class.fit(tr_data, validation_data=val_data, epochs=args.num_epochs, callbacks=calbacks, initial_epoch=10)
    else:
        # train without teacher signal
        model_class.fit(tr_data, validation_data=val_data, epochs=args.num_epochs, callbacks=calbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings to run CORNet')
    parser.add_argument('--v', default="None", help='visual info', choices=['V1', 'V4', 'IT', 'None'])
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--step', default=100, type=int, help='n steps to do v update')
    parser.add_argument('--r', default=0.01, type=float, help='ratio of visual data to run')
    parser.add_argument('--img_size', default=227, type=int, help='size to crop cif100 images')
    parser.add_argument('--batch_size', default=128, type=int, help='the batch size')
    parser.add_argument('--v_batch_size', default=128, type=int, help='the v batch size')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='the drop out factor')
    parser.add_argument('--num_classes', default=100, type=int, help='the drop out factor')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--r_per_epoch', default=False, action='store_true')
    parser.add_argument('--crop', default=False, action='store_true', help='crop the monkey images')
    parser.add_argument('--no-log', default=False, action='store_true', help='omit output (for testing)')
    parser.add_argument('--output', default='.', help='the output directory')
    argp = parser.parse_args()

    run_net(argp)
