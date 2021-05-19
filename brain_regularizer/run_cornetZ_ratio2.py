import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

from brain_regularizer.regularizer import RSA
from brain_regularizer.brain_data import MonkeyV1
from brain_regularizer.data_set import Cifar100
from brain_regularizer.network import cornet_zv
from brain_regularizer.helper import getCallbacks, ScaleR


def run_net(args):
    print(args)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    train_data, validation_data = Cifar100(img_size=args.img_size, batch_size=args.batch_size, scale=args.scale)
    net = cornet_zv(train_data, args.dropout_rate, args.learning_rate)
    print(net.summary())

    callbacks = getCallbacks(args)

    if args.v != 'None':
        brain_data = MonkeyV1(batch_size=1)
        joined_net, joined_train_data, joined_validation_data = RSA(net, "v1", train_data, validation_data, brain_data)
        callback_scale = ScaleR(joined_train_data, joined_net, args.r, args.r_per_epoch)
        joined_net.compile(optimizer=net.optimizer,
                           loss={"class": net.loss, "rsa_lambda": "mean_absolute_error"},
                           metrics=["accuracy"])

        # the initial epochs with teacher
        joined_net.fit(joined_train_data, validation_data=joined_validation_data, shuffle=False, epochs=10, callbacks=[callback_scale]+callbacks)
        # the rest of the epochs without a teacher
        net.fit(train_data, validation_data=validation_data, epochs=args.num_epochs, initial_epoch=10, callbacks=callbacks)
    else:
        # train without teacher signal
        net.fit(train_data, validation_data=validation_data, epochs=args.num_epochs, callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings to run CORNet')
    parser.add_argument('--v', default="V1", help='visual info', choices=['V1', 'V4', 'IT', 'None'])
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
    parser.add_argument('--v_center', default=False, action='store_true', help='whether to substract the mean of the monkey images')
    parser.add_argument('--v_scale', default=False, action='store_true', help='whether to scale of the monkey images')
    parser.add_argument('--scale', default=False, action='store_true', help='whether to scale of the cifar100 images')
    parser.add_argument('--no-log', default=False, action='store_true', help='omit output (for testing)')
    parser.add_argument('--output', default='logs', help='the output directory')
    argp = parser.parse_args()

    run_net(argp)
