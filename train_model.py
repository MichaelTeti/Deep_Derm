from argparse import ArgumentParser
from data_utils import *
from glob import glob
import os

############################### Arguments #####################################
parser = ArgumentParser()
parser.add_argument(
    'train_directory_neg',
    type=str,
    help='Where to look for negative training input images. Required.')
parser.add_argument(
    'train_directory_pos',
    type=str,
    help='Where to look for positive training input images. Required.')
parser.add_argument(
    '--model',
    type=str,
    choices=['vgg16', 'inception_v3', 'resnet50', 'mobilenet', 'nasnet', 'xception'],
    default='resnet50',
    help='Which model to train. Default is ResNet-50.')
parser.add_argument(
    '--image_size',
    type=int,
    default=200,
    help='The size to make each input square. Default is 500.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size to use for training. Default 200.')
parser.add_argument(
    '--positive_weight',
    type=float,
    default=10.,
    help='How much more to weight the positive samples than negative ones. Default 10.')
parser.add_argument(
    '--training_iters',
    type=int,
    default=5000,
    help='Number of training iterations. Default 5000.')
parser.add_argument(
    '--test_interval',
    type=int,
    default=200,
    help='How often to check validation performance. Default 200 iterations.')
###############################################################################

args = parser.parse_args()
train_dir_neg = args.train_directory_neg
train_dir_pos = args.train_directory_pos
model_name = args.model
im_size = args.image_size
batch_size = args.batch_size

# get filenames of all training images
filenames = glob(os.path.join(train_dir_neg, '*'))
filenames += glob(os.path.join(train_dir_pos, '*'))
filenames, test_filenames = split_training_and_testing(filenames)

if __name__ == '__main__':
    model = load_model(model_name, im_size)
    freeze_weights(model, model_name)
    model = add_trainable_layers(model, model_name)
    model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.Adam(lr=1e-3),
          metrics=['acc'])

    for iter in range(args.training_iters):
        data_batch, data_labels = load_batch(batch_size, filenames, im_size)
        train_hist = model.train_on_batch(data_batch,
                                    data_labels,
                                    class_weight={0: 1., 1: args.positive_weight})

        if iter % args.test_interval == 0:
            val_data, val_labels = load_batch(len(test_filenames), test_filenames, im_size)
            val_hist = model.test_on_batch(val_data, val_labels, sample_weight=None)
            print('Iteration: {}; Val. Loss: {}; Val. Acc.: {}'.format(iter, val_hist[0], val_hist[1]))
