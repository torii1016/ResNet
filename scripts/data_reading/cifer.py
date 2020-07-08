# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import pickle

class CiferData(object):
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path


    def _get_input_data(self, filename, rows, cols, channels, classnum):

        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding="bytes")
    
        data = dict[b'data']
        labels = np.array(dict[b'labels'])

        if labels.shape[0] != data.shape[0]:
            raise Exception('Error: Different length')
        num_images = labels.shape[0]

        data = data.reshape(num_images, channels, rows, cols)
        data = data.transpose([0,2,3,1])
        data = np.multiply(data, 1.0/255.0)

        labels = self._dense_to_one_hot(labels, classnum)

        return data, labels


    def _dense_to_one_hot(self, labels_dense, num_classes):

        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot


    def _input_data(self, label_id):

        images1, labels1 = self._get_input_data(self.data_folder_path+"data_batch_1", 32, 32, 3, 10)
        images2, labels2 = self._get_input_data(self.data_folder_path+"data_batch_2", 32, 32, 3, 10)
        images3, labels3 = self._get_input_data(self.data_folder_path+"data_batch_3", 32, 32, 3, 10)
        images4, labels4 = self._get_input_data(self.data_folder_path+"data_batch_4", 32, 32, 3, 10)
        images5, labels5 = self._get_input_data(self.data_folder_path+"data_batch_5", 32, 32, 3, 10)
        test_images, test_labels = self._get_input_data(self.data_folder_path+"test_batch", 32, 32, 3, 10)

        images = np.concatenate((images1, images2, images3, images4, images5), axis=0)
        labels = np.concatenate((labels1, labels2, labels3, labels4, labels5), axis=0)

        one_hot = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        one_hot[label_id] = 1.

        num = np.count_nonzero(labels==one_hot, axis=0)[label_id] 

        target_images = np.empty((num, 32, 32, 3))
        target_labels = np.empty((num, 10))

        num = 0
        for i in range(images.shape[0]):
            if labels[i][label_id]==1.0:
                target_images[num] = images[i]
                target_labels[num] = labels[i]
                num += 1

        return target_images, target_labels


    def describe_features(self, cnn, sess, label_id, output_start_id=0, output_end_id=-1):

        images, labels = self._input_data(label_id)

        features = cnn.get_feature(sess, images)[0]
        weights = np.ones(features.shape[0])*(1/features.shape[0])

        return features[output_start_id:output_end_id], weights[output_start_id:output_end_id]


    def training_data(self, varidation_size):

        images1, labels1 = self._get_input_data(self.data_folder_path+"data_batch_1", 32, 32, 3, 10)
        images2, labels2 = self._get_input_data(self.data_folder_path+"data_batch_2", 32, 32, 3, 10)
        images3, labels3 = self._get_input_data(self.data_folder_path+"data_batch_3", 32, 32, 3, 10)
        images4, labels4 = self._get_input_data(self.data_folder_path+"data_batch_4", 32, 32, 3, 10)
        images5, labels5 = self._get_input_data(self.data_folder_path+"data_batch_5", 32, 32, 3, 10)
        test_images, test_labels = self._get_input_data(self.data_folder_path+"test_batch", 32, 32, 3, 10)

        images = np.concatenate((images1, images2, images3, images4, images5), axis=0)
        labels = np.concatenate((labels1, labels2, labels3, labels4, labels5), axis=0)

        validation_images = images[:varidation_size]
        validation_labels = labels[:varidation_size]
        train_images = images[varidation_size:]
        train_labels = labels[varidation_size:]

        return train_images, train_labels, validation_images, validation_labels, test_images, test_labels