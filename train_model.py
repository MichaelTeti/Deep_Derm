from argparse import ArgumentParser
from glob import glob
import os
from data_utils import *
import csv
from sklearn.metrics import log_loss
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

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
    choices=['vgg16',
             'inception_v3',
             'resnet50',
             'mobilenet',
             'nasnet',
             'xception',
             'densenet',
             'senet154',
             'se_resnet50',
             'se_resnet101',
             'se_resnext50_32x4d',
             'se_resnext101_32x4d',
             'polynet',
             'resnext101_32x4d',
             'resnext101_64x4d',
             'resnet152'],
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
if args.brightness_shift_range is not None:
    bright_range = [int(num) for num in args.brightness_shift_range.split(',')]
else:
    bright_range = None
save_name = 'learning_rate_{}_positive_weight_{}_model_{}_batchsize_{}' \
                                            .format(args.learning_rate,
                                                    args.positive_weight,
                                                    args.model,
                                                    batch_size)

if model_name in ['senet154',
                  'se_resnet50',
                  'se_resnet101',
                  'se_resnext50_32x4d',
                  'se_resnext101_32x4d',
                  'polynet',
                  'resnext101_32x4d',
                  'resnext101_64x4d',
                  'resnet152']:
    framework = 'pt'
    dim_order = 'channels_first'
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils import data
    import torch.nn.functional as f
    import torch
else:
    framework = 'keras'
    dim_order = 'channels_last'
    import keras.backend as K
    from keras.models import model_from_json
    from keras.models import load_model

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

    model = load_pretrained_model(model_name, im_size, framework)
    freeze_weights(model, model_name, framework)
    model = add_trainable_layers(model, model_name, framework)

    if framework == 'keras':
        model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=args.learning_rate),
            metrics=['acc', tpr, fnr])
    elif framework == 'pt':
        class_weights = torch.from_numpy(np.array([1., args.positive_weight]))
        loss_func = nn.NLLLoss(weight=class_weights.cuda().float()) # define loss function
        opt = optim.Adam(model.parameters(), lr=args.learning_rate) # define optimizer

    # instantiate the training image generator and iterator
    train_gen = create_img_gen(args.flipud,
                               args.fliplr,
                               .15,
                               .15,
                               args.rotation_range,
                               bright_range,
                               dim_order)
    calculate_norm_coeff(train_gen, args.train_dir, im_size, framework)
    train_iterator = train_gen.flow_from_directory(
                     args.train_dir,
                     target_size=(im_size, im_size),
                     class_mode='categorical',
                     batch_size=batch_size)

    # instantiate the validation image generator and iterator
    val_gen = create_img_gen(False, False, None, None, 0, None, dim_order)
    calculate_norm_coeff(val_gen, args.train_dir, im_size, framework)
    val_iterator = val_gen.flow_from_directory(
                   args.validation_dir,
                   target_size=(im_size, im_size),
                   class_mode='categorical',
                   batch_size=batch_size,
                   shuffle=False)


    if framework == 'keras':
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
    else:
        for epoch in range(args.n_epochs):
            model.eval()  # put model in evaluation mode for validation
            val_loss_, Metrics = 0., np.array([0., 0., 0.])
            for val_iter in range(len(val_iterator)):
                val_data, val_labels = val_iterator.next()

                val_data = Variable(torch.from_numpy(val_data).cuda().float())
                val_labels = np.argmax(val_labels, 1)
                val_labels = Variable(torch.from_numpy(val_labels).cuda().long())

                val_output = f.log_softmax(model(val_data), dim=1)
                val_loss = loss_func(val_output, val_labels)

                metrics = TPR_FNR(val_labels, val_output)
                Metrics += metrics
                val_loss_ += val_loss.data.cpu().numpy()



            model.train() # put model in train model for training
            for iter in range(len(train_iterator)):
                train_data, train_labels = train_iterator.next()

                # prepare data for pytorch model on gpu
                train_data = Variable(torch.from_numpy(train_data).cuda().float())
                train_labels = np.argmax(train_labels, 1)
                train_labels = Variable(torch.from_numpy(train_labels).cuda().long())

                output = f.log_softmax(model(train_data), dim=1)
                loss = loss_func(output, train_labels)

                opt.zero_grad() # zero gradient for this iteration
                loss.backward() # accumulate gradient
                opt.step() # take a step downhill


            Metrics /= (val_iter + 1)
            val_loss_ /= (val_iter + 1)
            Metrics = np.round(Metrics, 2)
            val_loss_ = np.round(val_loss_, 2)
            print('Epoch: {}; Val. Acc.: {}; Val. TPR: {}; \
                  Val. FNR: {}; Val. Loss: {}'.format(epoch,
                                                      Metrics[0],
                                                      Metrics[1],
                                                      Metrics[2],
                                                      val_loss_))

            model_save_name = os.path.join(save_name, 'epoch{}.pt'.format(epoch))
            torch.save(model.state_dict(), model_save_name)



elif args.mode == 'test':
    args.test_dir = os.path.abspath(args.test_dir)
    assert(args.model_location is not None), 'No model location given to use.'
    args.model_location = os.path.abspath(args.model_location)
    batch_size = 1

    if framework == 'keras':
        # load the model
        weights_files = glob(os.path.join(args.model_location, '*.hdf5'))
        weights_files.sort()
        model = load_model(weights_files[-1],
                           custom_objects={'tpr': tpr, 'fnr': fnr})
    else:
        weights_files = glob(os.path.join(args.model_location, '*.pt'))
        weights_files.sort()
        model = load_pretrained_model(model_name, im_size, framework)
        model = add_trainable_layers(model, model_name, framework)
        model.load_state_dict(torch.load(weights_files[-1]))
        model = model.cuda()
        model.eval()

    test_gen = create_img_gen(False, False, None, None, 0, None, dim_order)
    calculate_norm_coeff(test_gen, os.path.abspath(args.train_dir), im_size, framework)
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
        #test_data = test_gen.standardize(test_data)
        if framework == 'keras':
            outputs.append(model.predict_on_batch(test_data)[0, 1])
        else:
            test_data = Variable(torch.from_numpy(test_data).cuda().float(), requires_grad=False)
            output = f.softmax(model(test_data), dim=1)
            outputs.append(output.detach().cpu().numpy()[0, 1])

    name_and_output = list(zip(file_names, outputs))
    name_and_output.sort(key=lambda elem: elem[0])

    for row, info in enumerate(name_and_output):
        writer.writerow([info[0], info[1]])
