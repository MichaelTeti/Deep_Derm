import numpy as np
from numpy.random import randint, randn
from scipy.misc import imsave, imshow, bytescale, imread, imresize
from glob import glob
from torch.autograd import Variable
import os
from progressbar import ProgressBar

try:
    import pretrainedmodels
except ImportError:
    os.system('pip3 install pretrainedmodels')
    import pretrainedmodels

import torch
import torch.nn as nn



def load_pretrained_model(model_name, image_size):
    model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
    model.input_size = (3, image_size[0], image_size[1])

    return model



def freeze_weights(model, model_name):
    model.cuda()
    for param in model.parameters():
        param.requires_grad=False

    return model



def add_trainable_layers(model, model_name, n_classes):
    if hasattr(model, 'avgpool'):
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1)).cuda()
    elif hasattr(model, 'avg_pool'):
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).cuda()

    input_feats = model.last_linear.in_features  # number of inputs to output layer
    model.last_linear = nn.Linear(input_feats, n_classes).cuda()  # take off last layer and add new one

    return model



def TPR_FNR(y_true, y_pred):
    y_pred_pos = np.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(y_true)
    y_neg = 1 - y_pos
    tp = np.sum(y_pos * y_pred_pos) / (np.sum(y_pos) + 1e-10)
    fn = np.sum(y_pos * y_pred_neg) / (np.sum(y_pos) + 1e-10)
    acc = np.mean(y_true == y_pred)

    return acc, tp, fn



def validate_model(model, val_iterator, loss_func, epoch):
    n_samples = val_iterator.samples
    val_loss_, Metrics = 0., np.zeros([n_samples, 2])
    bar = ProgressBar()

    print('Performing Validation...')
    for val_iter in bar(range(len(val_iterator))):
        val_data, val_labels = val_iterator.next()

        val_data = Variable(torch.from_numpy(val_data).cuda().float())
        bs = val_data.shape[0]

        val_labels = np.argmax(val_labels, 1)
        val_labels = Variable(torch.from_numpy(val_labels).cuda().long())

        val_output = model(val_data)
        val_loss = loss_func(val_output, val_labels)

        #metrics = TPR_FNR(val_labels, val_output)
        output_save = np.argmax(val_output.data.cpu().numpy(), 1)
        labels_save = val_labels.data.cpu().numpy()

        Metrics[val_iter*bs:val_iter*bs+bs, 0] = output_save
        Metrics[val_iter*bs:val_iter*bs+bs, 1] = labels_save
        val_loss_ += val_loss.data.cpu().numpy()

    Metrics = TPR_FNR(Metrics[:, 1], Metrics[:, 0])
    Metrics = np.round(Metrics, 2)

    val_loss_ /= (val_iter + 1)
    val_loss_ = np.round(val_loss_, 2)

    if val_output.size(1) == 2:
        print('Epoch: {}; Val. Acc.: {}; Val. TPR: {}; \
            Val. FNR: {}; Val. Loss: {}'.format(epoch,
                                                Metrics[0],
                                                Metrics[1],
                                                Metrics[2],
                                                val_loss_))
    else:
        print('Epoch: {}; Val. Acc.: {}; Val. TPR: {}; \
            Val. FNR: {}; Val. Loss: {}'.format(epoch,
                                                Metrics[0],
                                                '---',
                                                '---',
                                                val_loss_))




def model_trainer(model, train_iterator, loss_func, opt, snapshot_step,
                  val_iterator, save_name, epoch):

    train_bar = ProgressBar()
    print('Training...')
    #model.train()
    for iter in train_bar(range(len(train_iterator))):
        if iter % snapshot_step == 0:
            model.eval()
            validate_model(model, val_iterator, loss_func, epoch)
            model.train()

        assert(model.train()), 'model was left in eval mode when training.'
        train_data, train_labels = train_iterator.next()

        # prepare data for pytorch model on gpu
        train_data = Variable(torch.from_numpy(train_data).cuda().float())
        train_labels = np.argmax(train_labels, 1)
        train_labels = Variable(torch.from_numpy(train_labels).cuda().long())

        output = model(train_data)
        loss = loss_func(output, train_labels)

        opt.zero_grad() # zero gradient for this iteration
        loss.backward() # accumulate gradient
        opt.step() # take a step downhill

    model_save_name = os.path.join(save_name, 'epoch{}.pt'.format(epoch))
    torch.save(model.state_dict(), model_save_name)
