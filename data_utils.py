import numpy as np
from random import random
from scipy.ndimage.interpolation import rotate
from numpy.random import randint, randn
import matplotlib.pyplot as plt
from scipy.misc import imsave, imshow, bytescale, imread, imresize
from skimage.transform import resize
import keras
import tensorflow as tf
from glob import glob
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras import models
from keras import layers
from keras import optimizers
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
import os
import sklearn
from secrets import choice
from sklearn.metrics import log_loss
from keras.applications import vgg16, \
                               inception_v3, \
                               resnet50, \
                               mobilenet, \
                               nasnet, \
                               xception, \
                               densenet


def create_img_gen(flipud, fliplr, ht_shift, wd_shift, rot_range, bright_shift):
    return ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True,
                              horizontal_flip=flipud,
                              vertical_flip=fliplr,
                              height_shift_range=ht_shift,
                              width_shift_range=wd_shift,
                              fill_mode='reflect',
                              validation_split=0.0,
                              rotation_range=rot_range,
                              brightness_range=bright_shift)


def calculate_norm_coeff(img_gen, filepath, img_size):
    samples = glob(os.path.join(filepath, 'Positive/*'))
    samples += glob(os.path.join(filepath, 'Negative/*'))

    imgs = [imresize(imread(choice(samples)), [img_size, img_size]) for _ in range(2000)]
    img_gen.fit(imgs, augment=True, rounds=2)



def load_pretrained_model(model_name, image_size):
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

    elif model_name == 'densenet':
        return densenet.DenseNet121(include_top=False,
                                    input_shape=(image_size, image_size, 3))


def freeze_weights(model, model_name, retrain_frac):
    if model_name != 'mobilenet':
        for idx, layer in enumerate(model.layers):
            #if len(model.layers) - idx > int(len(model.layers)*retrain_frac):
            if 'bn' not in layer.name:
                layer.trainable = False


def add_trainable_layers(model, model_name):
    new_model = models.Sequential()
    new_model.add(model)

    if model_name == 'vgg16':
        new_model.add(layers.Flatten())
        # if vgg too big, change 4096 to some smaller number
        new_model.add(layers.Dense(4096, activation='relu'))
        new_model.add(layers.Dropout(0.5))
        new_model.add(layers.Dense(4096, activation='relu'))
        new_model.add(layers.Dropout(0.5))
        new_model.add(layers.Dense(2, activation='softmax'))

    else:
        new_model.add(layers.GlobalAveragePooling2D())
        new_model.add(layers.Dropout(0.6))
        new_model.add(layers.Dense(2, activation='softmax', kernel_initializer='he_normal'))

    return new_model



def tpr(y_true, y_pred):
    y_pred_pos = K.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(y_true)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + 1e-10)
    fn = K.sum(y_pos * y_pred_neg) / (K.sum(y_pos) + 1e-10)
    return tp / (tp + fn + 1e-10)


def fnr(y_true, y_pred):
    y_pred_pos = K.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(y_true)
    y_neg = 1 - y_pos
    fn = K.sum(y_pos * y_pred_neg) / (K.sum(y_pos) + 1e-10)
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + 1e-10)
    return fn / (fn + tp + 1e-10)
