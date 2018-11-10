import numpy as np
from numpy.random import randint, randn
from scipy.misc import imsave, imshow, bytescale, imread, imresize
import keras
from glob import glob
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
from secrets import choice
from keras.applications import vgg16, \
                               inception_v3, \
                               resnet50, \
                               mobilenet, \
                               nasnet, \
                               xception, \
                               densenet

try:
    import pretrainedmodels
except ImportError:
    os.system('pip3 install pretrainedmodels')
    import pretrainedmodels

import torch
import torch.nn as nn


def create_img_gen(flipud, fliplr, ht_shift, wd_shift, rot_range, bright_shift, dim_order):
    return ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True,
                              horizontal_flip=flipud,
                              vertical_flip=fliplr,
                              height_shift_range=ht_shift,
                              width_shift_range=wd_shift,
                              fill_mode='reflect',
                              validation_split=0.0,
                              rotation_range=rot_range,
                              brightness_range=bright_shift,
                              data_format=dim_order)


def calculate_norm_coeff(img_gen, filepath, img_size, fw):
    samples = glob(os.path.join(filepath, 'Positive/*'))
    samples += glob(os.path.join(filepath, 'Negative/*'))

    imgs = [imresize(imread(choice(samples)), [img_size, img_size]) for _ in range(2000)]

    if fw == 'keras':
        img_gen.fit(imgs, augment=True, rounds=2)
    else:
        img_gen.fit(np.asarray(imgs).transpose((0, 3, 1, 2)), augment=True, rounds=2)



def load_pretrained_model(model_name, image_size, fw):
    if fw == 'keras':
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

    elif fw == 'pt':
        model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        model.input_size = (3, image_size, image_size)

        return model


def freeze_weights(model, model_name, fw):
    if fw == 'keras':
        if model_name != 'mobilenet':
            for idx, layer in enumerate(model.layers):
                if 'bn' not in layer.name:
                    layer.trainable = False
    elif fw == 'pt':
        model.cuda()
        for param in model.parameters():
            param.requires_grad=False


def add_trainable_layers(model, model_name, fw):
    if fw == 'keras':
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

    elif fw == 'pt':
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).cuda()
        input_feats = model.last_linear.in_features  # number of inputs to output layer
        model.last_linear = nn.Linear(input_feats, 2).cuda()  # take off last layer and add new one

        return model



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


def TPR_FNR(y_true, y_pred):
    y_true = y_true.data.cpu().numpy()
    y_pred = np.argmax(y_pred.data.cpu().numpy(), 1)
    y_pred_pos = np.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(y_true)
    y_neg = 1 - y_pos
    tp = np.sum(y_pos * y_pred_pos) / (np.sum(y_pos) + 1e-10)
    fn = np.sum(y_pos * y_pred_neg) / (np.sum(y_pos) + 1e-10)
    acc = np.mean(y_true == y_pred)

    return acc, tp, fn
