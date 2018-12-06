from argparse import ArgumentParser
from glob import glob
import os
from data_utils import *
from pixel_flipping_salience import *
import csv
from sklearn.metrics import log_loss
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as f
from keras.preprocessing.image import ImageDataGenerator
import torch

############################### Arguments #####################################
parser = ArgumentParser()

parser.add_argument(
    '--train_dir',
    type=str,
    default='Train',
    help='Where to look for training images.')
parser.add_argument(
    '--validation_dir',
    type=str,
    default='Validation',
    help='Where to look for validation input images.')
parser.add_argument(
    '--model_location',
    type=str,
    default=None,
    help='The directory where the saved model is located if testing.')
parser.add_argument(
    '--mode',
    type=str,
    choices=['train', 'test'],
    default='train',
    help='Whether to finetune a model or test a trained one.')
parser.add_argument(
    '--model',
    type=str,
    choices=['vgg11_bn',
             'vgg16',
             'vgg16_bn',
             'vgg19_bn',
             'vgg19',
             'resnet18',
             'resnet34',
             'resnet101',
             'resnet50',
             'resnet152',
             'squeezenet1_0',
             'squeezenet1_1',
             'densenet121',
             'densenet169',
             'densenet201',
             'fbresnet152',
             'resnext101_32x4d',
             'resnext101_64x4d',
             'inceptionv4',
             'inceptionresnetv2',
             'nasnetamobile',
             'senet154',
             'se_resnet50',
             'se_resnet101',
             'se_resnet152',
             'se_resnext50_32x4d',
             'se_resnext101_32x4d',
             'cafferesnet101',
             'polynet',
             'pnasnet5large'],
    default='resnet50',
    help='Which model to train. Default is ResNet-50.')
parser.add_argument(
    '--image_size',
    type=str,
    default='224,224',
    help='The size to make each input square. Default is 200x200.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=40,
    help='Batch size to use for training. Default 64.')
parser.add_argument(
    '--class_weights',
    type=str,
    default=None,
    help='How much more to weight each class in the loss function.')
parser.add_argument(
    '--class_weight_indices',
    type=str,
    default=None,
    help='The corresponding class indices for the class weights.')
parser.add_argument(
    '--n_epochs',
    type=int,
    default=250,
    help='Number of training epochs. Default 250.')
parser.add_argument(
    '--fliplr',
    type=bool,
    default=False,
    help='Augment the images by flipping some of them across y-axis randomly. Default False.')
parser.add_argument(
    '--flipud',
    type=bool,
    default=False,
    help='Augment images by flipping some of them across x-axis. Default False.')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=.0001,
    help='Learning rate for training. Default is 1e-4.')
parser.add_argument(
    '--lr_decay',
    type=float,
    default=0.2,
    help='Learning rate decay parameter.')
parser.add_argument(
    '--save_freq',
    type=int,
    default=200,
    help="How often in training iterations to view model's progress.")
parser.add_argument(
    '--rotation_range',
    type=int,
    default=0,
    help='Rotation angle in degrees when augmenting.')
parser.add_argument(
    '--brightness_shift_range',
    type=str,
    default=None,
    help='Amount to shift the brightness. E.g. 10,40')
###############################################################################

args = parser.parse_args()
model_name = args.model
im_size = [int(n) for n in args.image_size.split(',')]
batch_size = args.batch_size
args.train_dir = os.path.abspath(args.train_dir)
args.validation_dir = os.path.abspath(args.validation_dir)

if args.brightness_shift_range is not None:
    bright_range = [int(num) for num in args.brightness_shift_range.split(',')]
else:
    bright_range = None

save_name = 'learning_rate_{}_model_{}_batchsize_{}' \
                                            .format(args.learning_rate,
                                                    args.model,
                                                    batch_size)

# make a new folder for the results of this model if not one already
if save_name not in os.listdir():
    os.mkdir(save_name)


def normalize(input_image):
    if np.amin(input_image) < 0.:
        input_image += np.amin(input_image)
    elif np.amin(input_image) > 0.:
        input_image -= np.amin(input_image)

    input_image /= (np.amax(input_image) + 1e-10)

    if in_range[-1] == 255:
        input_image *= 255.

    for channel in range(3):
        input_image[channel, ...] -= mu[channel]
        input_image[channel, ...] /= sigma[channel]

    return input_image


def create_img_gen(flipud, fliplr, ht_shift, wd_shift, rot_range, bright_shift):
    return ImageDataGenerator(horizontal_flip=flipud,
                              vertical_flip=fliplr,
                              height_shift_range=ht_shift,
                              width_shift_range=wd_shift,
                              fill_mode='reflect',
                              validation_split=0.0,
                              rotation_range=rot_range,
                              brightness_range=bright_shift,
                              data_format='channels_first',
                              preprocessing_function=normalize)


if args.mode == 'train':
    # instantiate the training image generator and iterator
    train_gen = create_img_gen(args.flipud,
                               args.fliplr,
                               .15,
                               .15,
                               args.rotation_range,
                               bright_range)
    train_iterator = train_gen.flow_from_directory(
                     args.train_dir,
                     target_size=(im_size[0], im_size[1]),
                     class_mode='categorical',
                     batch_size=batch_size)

    # instantiate the validation image generator and iterator
    val_gen = create_img_gen(False, False, None, None, 0, None)
    val_iterator = val_gen.flow_from_directory(
                   args.validation_dir,
                   target_size=(im_size[0], im_size[1]),
                   class_mode='categorical',
                   batch_size=batch_size,
                   shuffle=False)

    n_classes = train_iterator.num_classes


    model = load_pretrained_model(model_name, im_size)
    model = freeze_weights(model, model_name)
    model = add_trainable_layers(model, model_name, n_classes)

    mu, sigma, in_range = model.mean, model.std, model.input_range

    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    class_weights = np.ones([n_classes, ])
    if args.class_weights is not None:
        class_penalties = [float(num) for num in args.class_weights.split(',')]
        class_indices = [int(num) for num in args.class_weight_indices.split(',')]
        assert(len(class_penalties) == len(class_indices)), 'num. penalties and indices not equal'

        for weight_indx in range(len(class_indices)):
            position = class_indices[weight_indx]
            class_weights[position] = class_penalties[weight_indx]


    loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).cuda().float()) # define loss function
    opt = optim.Adam(model.parameters(), lr=args.learning_rate) # define optimizer


    for epoch in range(args.n_epochs):
        model_trainer(model,
                      train_iterator,
                      loss_func,
                      opt,
                      args.save_freq,
                      val_iterator,
                      save_name,
                      epoch)

elif args.mode == 'test':
    args.validation_dir = os.path.abspath(args.validation_dir)
    n_classes = len(os.listdir(args.validation_dir))

    args.model_location = os.path.abspath(args.model_location)
    weights_files = glob(os.path.join(args.model_location, '*.pt'))
    model = load_pretrained_model(model_name, im_size)
    model = add_trainable_layers(model, model_name, n_classes)
    mu, sigma, in_range = model.mean, model.std, model.input_range
    model.load_state_dict(torch.load(weights_files[-1]))
    model = model.cuda()
    model.eval()

    pixy = PixelFlip(model, args.validation_dir, args.model_location)
    pixy.flip()
