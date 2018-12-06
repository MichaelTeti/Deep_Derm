import numpy as np
from torch.autograd import Variable
import torch
from scipy.misc import imresize, imread, imshow
import os
from glob import glob
from progressbar import ProgressBar
import matplotlib.pyplot as plt

class PixelFlip:
    def __init__(self, model, data_path, model_location):
        self.model = model
        self.model_location = model_location
        if 'Pixel Flip Results' not in os.listdir(self.model_location):
            os.mkdir(os.path.join(self.model_location, 'Pixel Flip Results'))

        self.model_location = os.path.join(self.model_location, 'Pixel Flip Results')
        self.folders = glob(os.path.join(data_path, '*'))

        for folder_idx, folder in enumerate(self.folders):
            if folder_idx == 0:
                self.filenames = glob(os.path.join(folder, '*'))
                continue
            else:
                self.filenames += glob(os.path.join(folder, '*'))

        self.n_test = len(self.filenames)
        self.img_h, self.img_w = self.model.input_size[1:]
        self.mse = np.zeros([self.n_test, self.img_h, self.img_w])
        self.n_flipped = np.zeros([self.n_test, self.img_h, self.img_w])
        self.mu = torch.from_numpy(np.array(self.model.mean)).cuda().float()
        self.sigma = torch.from_numpy(np.array(self.model.std)).cuda().float()


    def preprocess(self, image):
        # scale the values to the range [0, 1]
        min_val = torch.min(image)
        if min_val < 0.:
            image += min_val
        elif min_val > 0.:
            image -= min_val

        image /= (torch.max(image) + 1e-10)

        # normalize each channel

        for channel in range(3):
            image[:, channel, ...] -= self.mu[channel]
            image[:, channel, ...] /= self.sigma[channel]

        return image


    def flip(self):
        for data_num, test_sample in enumerate(self.filenames):
            img = imresize(imread(test_sample), [self.img_h, self.img_w])
            img_show = img
            img = img.transpose((2, 0, 1))
            img = Variable(torch.from_numpy(img[None, ...]).cuda().float())
            bar = ProgressBar()

            for row in bar(range(img.size(2))):
                for col in range(img.size(3)):
                    if row == 0 and col == 0:
                        original_output = self.model(self.preprocess(img)).data.cpu().numpy()
                        original_label = np.argmax(original_output, 1)

                    modified = img

                    for channel in range(modified.size(1)):
                        if modified[0, channel, row, col] >= 128:
                            modified[0, channel, row, col] = 0
                        else:
                            modified[0, channel, row, col] = 255

                    new_output = self.model(self.preprocess(modified)).data.cpu().numpy()
                    mse = np.mean((new_output - original_output)**2)
                    self.mse[data_num, row, col] = mse

                    if np.argmax(new_output, 1) != original_label:
                        self.n_flipped[data_num, row, col] = 255

            fig = plt.figure()
            subplot1 = fig.add_subplot(131)
            subplot1.set_title('Original Image')

            subplot2 = fig.add_subplot(132)
            subplot2.set_title('Output MSE')

            subplot3 = fig.add_subplot(133)
            subplot3.set_title('Flipped')

            subplot1.imshow(img_show)
            subplot2.imshow(self.mse[data_num, ...], cmap='gray')
            subplot3.imshow(self.n_flipped[data_num, ...], cmap='gray')
            plt.tight_layout()

            fig_name = os.path.split(test_sample)[-1]
            plt.savefig(os.path.join(self.model_location, fig_name))
