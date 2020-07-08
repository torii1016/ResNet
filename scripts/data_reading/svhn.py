# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
from scipy.io import loadmat

class SVHNData(object):
    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def get_test_data(self):
        return self._create_svhn_dataset(self.test_data_path)

    def get_train_data(self):
        return self._create_svhn_dataset(self.train_data_path)
    
    def _create_svhn_dataset(self, path):
        file = open(path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        labels = data['y'].flatten()
        labels[labels == 10] = 0
        labels = (np.arange(10) == labels[:, None]).astype(np.float32)
        img_array = self._convert_imgs_to_array(imgs)
        file.close()
        return img_array, labels

    def _convert_imgs_to_array(self, img_array):
        rows = img_array.shape[0]
        cols = img_array.shape[1]
        chans = img_array.shape[2]
        num_imgs = img_array.shape[3]
        scalar = 1 / 255
        new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
        for x in range(0, num_imgs):
            chans = img_array[:, :, :, x]
            norm_vec = (255-chans)*1.0/255.0
            new_array[x] = norm_vec
        return new_array
