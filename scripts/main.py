import argparse
import sys
import numpy as np
from scipy.io import loadmat
import cv2
from tqdm import tqdm

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from data_reading import CiferData
from model.resnet import ResNet
from model.convolutional_neural_network import ConvolutionalNeuralNetwork

#from show_data import ShowData

def create_svhn_dataset(path):
    file = open(path, 'rb')
    data = loadmat(file)
    imgs = data['X']
    labels = data['y'].flatten()
    labels[labels == 10] = 0
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)
    img_array = convert_imgs_to_array(imgs)
    file.close()
    return img_array, labels

def convert_imgs_to_array(img_array):
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

def color_resize(dataset):

    output_dataset = np.empty(shape=(dataset.shape[0], 32, 32, 3), dtype=np.float32)
    for i in range(output_dataset.shape[0]):
        data = dataset[i].reshape(28,28)
        tmp = np.zeros([28,28,3])
        tmp[:,:,0] = data
        tmp[:,:,1] = data
        tmp[:,:,2] = data

        tmp = cv2.resize(tmp, (32,32))
        output_dataset[i,:,:,:] = tmp
    return output_dataset

def main():

    svhn_train_data, svhn_train_label = create_svhn_dataset("Dataset/SVHN/train_32x32.mat")
    svhn_pre_train_data = svhn_train_data[:60000]
    svhn_pre_train_label = svhn_train_label[:60000]
    svhn_fine_tuning_data = svhn_train_data[svhn_train_data.shape[0]-10000:]
    svhn_fine_tuning_label = svhn_train_label[svhn_train_data.shape[0]-10000:]
    svhn_test_data, svhn_test_label = create_svhn_dataset("Dataset/SVHN/test_32x32.mat")
    svhn_pre_test_data = svhn_train_data[6000:svhn_train_data.shape[0]-10000]
    svhn_pre_test_label = svhn_train_label[6000:svhn_train_data.shape[0]-10000]


    print("--- Loading dataset ---") # -------------------------------------
    cifer10 = CiferData("Dataset/CIFAR10/")

    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = cifer10.training_data(7000)
    # ----------------------------------------------------------------------

    train_data = train_images
    train_label = train_labels
    test_data = test_images
    test_label = test_labels

    resnet = ResNet(32, 10)
    #resnet = ConvolutionalNeuralNetwork(32, 10)
    resnet.set_model(0.001)

    #saver = tf.compat.v1.train.Saver
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    epoch = 1000
    batch_size = 50

    # Pre-training
    accuracy_list = []
    loss_list = []
    for i in tqdm(range(epoch)):
        choice_id = np.random.choice(train_data.shape[0], batch_size, replace=False)
        batch_data = train_data[choice_id]
        batch_label = train_label[choice_id]

        """
        if i % 10 == 0:
            accuracy = 0
            for j in range(0, test_data.shape[0], 100):
                data = test_data[j:j+100]
                label = test_label[j:j+100]
                accuracy += int(resnet.test(sess, data, label)[0]*data.shape[0])
            print("step {}, training accuracy {}".format(i, accuracy/test_data.shape[0]*100.0))
        """

        _, loss = resnet.train(sess, batch_data, batch_label)
        loss_list.append(loss)
        print("loss: {}".format(loss))

    accuracy = 0
    for j in range(0, test_data.shape[0], 100):
        data = test_data[j:j+100]
        label = test_label[j:j+100]
        accuracy += int(resnet.test(sess, data, label)[0]*data.shape[0])

    print("test accuracy {}".format(accuracy/test_data.shape[0]*100.0))

if __name__ == '__main__':
    main()