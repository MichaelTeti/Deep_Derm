import numpy as np
from random import random
from scipy.ndimage.interpolation import rotate
from numpy.random import randint, randn
from scipy.misc import imread, imresize, imsave, imshow
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import vgg16, \
                               inception_v3, \
                               resnet50, \
                               mobilenet, \
                               nasnet, \
                               xception


def split_training_and_testing(filenames):
    '''Splits the filenames into training (80%) and testing (20%).'''

    test_indx = randint(0, len(filenames), int(0.2*len(filenames)))
    test_fnames = [filenames[i] for i in test_indx]
    for index in sorted(test_indx, reverse=True):
        del filenames[index]

    return filenames, test_fnames


def load_batch(batch_size, filenames, img_size, fliplr, flipud, rand_crop, mode):
    '''Loads in a batch of training images each training iteration
       Args:
            batch_size, int: The number of images in each training batch

            filenames: A list of filenames pointing to the training images

            img_size, int: An integer describing the image height and width

            fliplr, bool: Whether or not to randomly flip some images across y-axis

            flipud, bool: Whether or not to randomly flip some images across x-axis

            rand_crop, bool: Whether or not to perform random cropping on images
    '''

    assert(batch_size <= len(filenames)), 'batch size bigger than number of files.'

    if mode == 'train':
        rand_files = randint(0, len(filenames), batch_size) # batch_size random indices
    elif mode == 'test':
        rand_files = list(range(len(filenames)))

    filenames_use = [filenames[i] for i in rand_files]  # get files given in rand_files
    labels = []  # create empty list to accumulate labels

    for ind, filename in enumerate(filenames_use):
        im_read = imread(filename) # read in each image

        # if the image shape is not the desired shape, resize it
        if im_read.shape[0] != img_size or im_read.shape[1] != img_size:
            im_read = imresize(im_read, (img_size, img_size))

        # image augmentation
        if fliplr and random() > 0.5:
            im_read = np.fliplr(im_read)
        if flipud and random() > 0.5:
            im_read = np.flipud(im_read)
        if rand_crop and random() > 0.5:
            pad = int(img_size * .1)
            im_read = np.pad(im_read, ((pad, pad), (pad, pad), (0, 0)), 'constant')
            r, c = randint(0, pad*2, 2)
            im_read = im_read[r:r+img_size, c:c+img_size, :]

        # add appropriate label to each image
        if 'train' in filename or 'Negative' in filename:
            labels.append([1, 0])
        elif 'test' in filename or 'Positive' in filename:
            labels.append([0, 1])

        # create empty array to accumulate images if reading in first image
        if ind == 0:
            if len(im_read.shape) == 2:
                images = np.zeros([batch_size, img_size, img_size])
            elif len(im_read.shape) == 3:
                images = np.zeros([batch_size, img_size, img_size, 3])

        images[ind, ...] = im_read  # add image to the rest of the images

    # normalization
    images = (images - np.mean(images, 0)) / (np.std(images, 0) + 1e-8)
    return images, np.asarray(labels)


def load_model(model_name, image_size):
    if model_name == 'vgg16':
        return vgg16.VGG16(include_top=False,
                           input_shape=(image_size, image_size, 3))

    elif model_name == 'inception_v3':
        return inception_v3.InceptionV3(include_top=False,
                                        input_shape=(image_size, image_size, 3))

    elif model_name == 'resnet50':
        return resnet50.ResNet50(include_top=False,
                                 input_shape=(image_size, image_size, 3))

    elif model_name == 'mobilenet':
        return mobilenet.MobileNet(include_top=False,
                                   input_shape=(image_size, image_size, 3),
                                   weights=None)

    elif model_name == 'nasnet':
        return nasnet.NASNetMobile(include_top=False,
                                   input_shape=(image_size, image_size, 3))

    elif model_name == 'xception':
        return xception.Xception(include_top=False,
                                 input_shape=(image_size, image_size, 3))


def freeze_weights(model, model_name):
    for layer in model.layers:
        if model_name != 'mobilenet':
            layer.trainable = False
    return


def add_trainable_layers(model, model_name):
    new_model = models.Sequential()
    new_model.add(model)

    if model_name == 'vgg16':
        new_model.add(layers.Flatten())
        new_model.add(layers.Dense(4096, activation='relu'))
        new_model.add(layers.Dropout(0.5))
        new_model.add(layers.Dense(4096, activation='relu'))
        new_model.add(layers.Dropout(0.5))
        new_model.add(layers.Dense(2, activation='softmax'))

    elif model_name == 'inception_v3':
        new_model.add(layers.GlobalAveragePooling2D())
        new_model.add(layers.Flatten())
        new_model.add(layers.Dropout(0.5))
        new_model.add(layers.Dense(2, activation='softmax'))

    elif model_name == 'resnet50':
        new_model.add(layers.GlobalAveragePooling2D())
        new_model.add(layers.Dense(2, activation='softmax'))

    elif model_name == 'mobilenet':
        new_model.add(layers.GlobalAveragePooling2D())
        new_model.add(layers.Dense(2, activation='softmax'))

    elif model_name == 'nasnet':
        new_model.add(layers.GlobalAveragePooling2D())
        new_model.add(layers.Dense(2, activation='softmax'))

    elif model_name == 'xception':
        new_model.add(layers.GlobalAveragePooling2D())
        new_model.add(layers.Dense(2, activation='softmax'))

    return new_model
