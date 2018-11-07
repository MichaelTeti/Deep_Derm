from argparse import ArgumentParser
from glob import glob
import os
from data_utils import *
from keras.models import model_from_json
import csv
from progressbar import ProgressBar
import keras.backend as K
from sklearn.metrics import log_loss
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import urllib.parse
from keras.models import load_model
import cv2
from cv2 import resize

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
    '--test_dir',
    type=str,
    default='Test',
    help='The directory where the test images are.')
parser.add_argument(
    '--model_location',
    type=str,
    default=None,
    help='The directory where the saved model is located if testing.')
parser.add_argument(
    '--mode',
    choices=['train', 'test'],
    type=str,
    default='train',
    help='Whether to train a model or test one that has already been trained.')

parser.add_argument(
    '--model',
    type=str,
    choices=['vgg16', 'inception_v3', 'resnet50', 'mobilenet', 'nasnet', 'xception', 'densenet'],
    default='resnet50',
    help='Which model to train. Default is ResNet-50.')
parser.add_argument(
    '--image_size',
    type=int,
    default=224,
    help='The size to make each input square. Default is 200x200.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=40,
    help='Batch size to use for training. Default 40.')
parser.add_argument(
    '--positive_weight',
    type=float,
    default=10.,
    help='How much more to weight the malignant samples than negative ones. Default 10.')
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
    default=.00001,
    help='Learning rate for training. Default is 1e-3.')
parser.add_argument(
    '--lr_decay',
    type=float,
    default=0.2,
    help='Learning rate decay parameter.')
parser.add_argument(
    '--save_freq',
    type=int,
    default=2,
    help="How often in epochs to save the model's progress.")
parser.add_argument(
    '--retrain_num',
    type=float,
    default=0.33,
    help="What fraction of the model's layers to retrain. Default 0.33.")
parser.add_argument(
    '--rotation_range',
    type=int,
    default=25,
    help='Rotation angle in degrees when augmenting.')
parser.add_argument(
    '--brightness_shift_range',
    type=str,
    default=None,
    help='Amount to shift the brightness. E.g. 10,40')
###############################################################################

args = parser.parse_args()
model_name = args.model
im_size = args.image_size
batch_size = args.batch_size
bright_range = [int(num) for num in args.brightness_shift_range.split(',')]
save_name = 'learning_rate_{}_positive_weight_{}_model_{}_batchsize_{}' \
                                            .format(args.learning_rate,
                                                    args.positive_weight,
                                                    args.model,
                                                    batch_size)

# tensorboard callback
tensorboard = TensorBoard(log_dir=os.path.join(os.getcwd(), 'tensorboard_logs/{}'.format(save_name)),
                     batch_size=batch_size,
                     write_graph=False,
                     update_freq='epoch')

# model_checkpoint callback
model_save_name = os.path.join(save_name, 'weights_epoch_{epoch:03d}.hdf5')
model_checkpoint = ModelCheckpoint(model_save_name,
                                   monitor='val_loss',
                                   save_best_only=False,
                                   period=args.save_freq)

# learning rate scheduler
def lr_decay(epoch, lr):
    if epoch > 49 and epoch < 125 and epoch % 25 == 0:
        return lr * args.lr_decay
    else:
        return lr

lr_sched = LearningRateScheduler(lr_decay, verbose=0)


# start tensorboard
os.system('tensorboard --logdir={} &'.format(os.path.join(os.getcwd(), 'tensorboard_logs')))

# make a new folder for the results of this model if not one already
if save_name not in os.listdir():
    os.mkdir(save_name)


if args.mode == 'train':
    args.train_dir = os.path.abspath(args.train_dir)
    args.validation_dir = os.path.abspath(args.validation_dir)

    model = load_pretrained_model(model_name, im_size)
    freeze_weights(model, model_name, args.retrain_num)
    model = add_trainable_layers(model, model_name)
    model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.Adam(lr=args.learning_rate),
          metrics=['acc', tpr, fnr])

    # instantiate the training image generator and iterator
    train_gen = create_img_gen(args.flipud,
                               args.fliplr,
                               .15,
                               .15,
                               args.rotation_range,
                               bright_range)
    calculate_norm_coeff(train_gen, args.train_dir, im_size)
    train_iterator = train_gen.flow_from_directory(
                     args.train_dir,
                     target_size=(im_size, im_size),
                     class_mode='categorical',
                     batch_size=batch_size)

    # instantiate the validation image generator and iterator
    val_gen = create_img_gen(False, False, None, None, 0, None)
    calculate_norm_coeff(val_gen, args.train_dir, im_size)
    val_iterator = val_gen.flow_from_directory(
                   args.validation_dir,
                   target_size=(im_size, im_size),
                   class_mode='categorical',
                   batch_size=batch_size,
                   shuffle=False)


    # perform training
    model.fit_generator(train_iterator,
                        epochs=args.n_epochs,
                        class_weight={0: 1., 1: args.positive_weight},
                        validation_data=val_iterator,
                        steps_per_epoch=train_iterator.__len__(),
                        validation_steps=val_iterator.__len__(),
                        verbose=2,
                        callbacks=[tensorboard, model_checkpoint, lr_sched],
                        workers=2,
                        use_multiprocessing=True)


elif args.mode == 'test':
    args.test_dir = os.path.abspath(args.test_dir)
    assert(args.model_location is not None), 'No model location given to use.'
    args.model_location = os.path.abspath(args.model_location)
    batch_size = 1

    # load the model
    weights_files = glob(os.path.join(args.model_location, '*.hdf5'))
    weights_files.sort()
    model = load_model(weights_files[-1],
                       custom_objects={'tpr': tpr, 'fnr': fnr})

    test_gen = create_img_gen(False, False, None, None, 0, None)
    calculate_norm_coeff(test_gen, os.path.abspath(args.train_dir), im_size)
    test_iterator = test_gen.flow_from_directory(
                    args.test_dir,
                    target_size=(im_size, im_size),
                    class_mode='categorical',
                    batch_size=batch_size,
                    shuffle=False)

    file_names = [os.path.split(f)[1] for f in test_iterator.filenames]
    outputs = []

    os.chdir(args.model_location)
    csvfile = open('results.csv', 'a')
    writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for test_num in range(test_iterator.__len__()):
        test_data, _ = test_iterator.next()
        test_data = test_gen.standardize(test_data)
        outputs.append(model.predict_on_batch(test_data)[0, 1])

    name_and_output = list(zip(file_names, outputs))
    name_and_output.sort(key=lambda elem: elem[0])

    for row, info in enumerate(name_and_output):
        writer.writerow([info[0], info[1]])
