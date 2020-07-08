import argparse
import sys
import numpy as np
from scipy.io import loadmat
import cv2
from tqdm import tqdm
from collections import OrderedDict
import toml

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from data_reading import CiferData, SVHNData
from model.resnet import ResNet50, ResNet18
from model.convolutional_neural_network import ConvolutionalNeuralNetwork

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


def main(config_dict):

    print("--- Loading dataset [{}] ---".format(config_dict["Dataset"]["name"])) # --------
    if config_dict["Dataset"]["name"] == "SVHN":
        svhn = SVHNData("Dataset/SVHN/train_32x32.mat", "Dataset/SVHN/test_32x32.mat")
        train_images, train_labels = svhn.get_train_data()
        test_images, test_labels = svhn.get_test_data()
    elif config_dict["Dataset"]["name"] == "CIFAR10":
        cifer10 = CiferData("Dataset/CIFAR10/")
        train_images, train_labels, validation_images, validation_labels, test_images, test_labels = cifer10.training_data(7000)
    else:
        print("Not dataset. please check the toml file")
        exit()
    # -------------------------------------------------------------------------------------

    train_data = train_images
    train_label = train_labels
    test_data = test_images
    test_label = test_labels

    print("--- Creating model [{}] ---".format(config_dict["Network"]["name"])) # ---------
    if config_dict["Network"]["name"] == "ResNet50":
        network = ResNet50(config_dict["Network"]["fig_size"], config_dict["Network"]["class"])
    elif config_dict["Network"]["name"] == "ResNet18":
        network = ResNet18(config_dict["Network"]["fig_size"], config_dict["Network"]["class"])
    else:
        network = ConvolutionalNeuralNetwork(config_dict["Network"]["fig_size"], config_dict["Network"]["class"])
    network.set_model(config_dict["Network"]["lr"])
    # -------------------------------------------------------------------------------------

    #saver = tf.compat.v1.train.Saver
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    epoch = config_dict["episode"]
    batch_size = config_dict["batch_size"]

    # train
    accuracy_list = []
    loss_list = []
    with tqdm(range(epoch)) as pbar:
        for i, ch in enumerate(pbar):
            choice_id = np.random.choice(train_data.shape[0], batch_size, replace=False)
            batch_data = train_data[choice_id]
            batch_label = train_label[choice_id]

            _, loss = network.train(sess, batch_data, batch_label)
            loss_list.append(loss)
            pbar.set_postfix(OrderedDict(loss=loss))

    # test
    accuracy = 0
    for j in range(0, test_data.shape[0], 100):
        data = test_data[j:j+100]
        label = test_label[j:j+100]
        accuracy += int(network.test(sess, data, label)[0]*data.shape[0])

    print("test accuracy {}".format(accuracy/test_data.shape[0]*100.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/config.toml", type=str, help="default: config/config.toml")
    args = parser.parse_args()

    main(toml.load(open(args.config)))