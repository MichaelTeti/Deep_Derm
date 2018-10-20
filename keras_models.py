import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import vgg16, \
                               inception_v3, \
                               resnet50, \
                               mobilenet, \
                               nasnet, \
                               xception, \
                               inception_resnet_v2


def load_model(model_name, data_shape):
    if model_name == 'vgg16':
        return vgg16.VGG16(include_top=False,
                           input_shape=data_shape)

    elif model_name == 'inception_v3':
        return inception_v3.InceptionV3(include_top=False,
                                        input_shape=data_shape)

    elif model_name == 'resnet50':
        return resnet50.ResNet50(include_top=False,
                                 input_shape=data_shape)

    elif model_name == 'mobilenet':
        return mobilenet.MobileNet(include_top=False,
                                   input_shape=data_shape,
                                   weights=None)

    elif model_name == 'nasnet':
        return nasnet.NASNetMobile(include_top=False,
                                   input_shape=data_shape)

    elif model_name == 'xception':
        return xception.Xception(include_top=False,
                                 input_shape=data_shape)

    elif model_name == 'inception_resnet_v2':
        return inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                     input_shape=data_shape)



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

    else:
        new_model.add(layers.GlobalAveragePooling2D())
        new_model.add(layers.Dense(2, activation='softmax'))

    return new_model
