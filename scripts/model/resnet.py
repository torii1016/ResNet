# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim, get_channel
from .residual_block import ResidualBlock

class _cnn(Layers):
    def __init__(self, name_scopes, layer_channels, filter_size):
        #assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.name_scope = name_scopes
        self.layer_channels = layer_channels
        self.filter_size = filter_size

    def set_model(self, inputs, is_training = True, reuse = False):

        h  = inputs
        # convolution
        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            for i, out_chan in enumerate(self.layer_channels):
                conved = conv(inputs = h,
                    out_num = out_chan,
                    filter_width = self.filter_size[i], filter_height = self.filter_size[i],
                    stride = 2, name=i)

                bn_conved = batch_norm(i, conved, is_training)
                h = tf.nn.relu(bn_conved)

        feature_image = h
        h = tf.nn.max_pool2d(h, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
        return h


class _nn(Layers):
    def __init__(self, name_scopes, layer_channels):
        #assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.name_scope = name_scopes
        self.layer_channels = layer_channels

    def set_model(self, inputs, is_training = True, reuse = False):

        h = self.global_average_pooling(inputs)

        dim = get_dim(h)
        h = tf.reshape(h, [-1, dim])

        # convolution
        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            for i, s in enumerate(self.layer_channels):
                lin = linear(i, h, s)
                h = lrelu(lin)

        return lin

    def global_average_pooling(self, x):
        for _ in range(2):
            x = tf.reduce_mean(x, axis=1)
        return x


class ResNet50(object):
    
    def __init__(self, fig_size, output_dim):
        self.cnn = _cnn('ResNetCNN', [64], [7])
        self.conv2_networks = [ResidualBlock('conv2_{}block'.format(i+1),
                                            [64, 64, 256],
                                            [1, 3, 1], None) for i in range(3)]

        self.conv3_networks = [ResidualBlock('conv3_{}block'.format(i+1),
                                            [128, 128, 512],
                                            [1, 3, 1], None) for i in range(4)]

        self.conv4_networks = [ResidualBlock('conv4_{}block'.format(i+1),
                                            [256, 256, 1024],
                                            [1, 3, 1], None) for i in range(6)]

        self.conv5_networks = [ResidualBlock('conv5_{}block'.format(i+1),
                                            [512, 512, 2048],
                                            [1, 3, 1], None) for i in range(3)]

        self.nn = _nn('ResNetNN', [1000, output_dim])
        self.fig_size = fig_size
        self.output_dim = output_dim
        
    def set_model(self, lr):

        self.lr = tf.Variable(
            name = "learning_rate",
            initial_value = lr,
            trainable = False)

        self.lr_op = tf.compat.v1.assign(self.lr, 0.95 * self.lr)

        # -- place holder ---
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self.fig_size, self.fig_size, 3])
        self.target_val = tf.compat.v1.placeholder(tf.float32, [None, self.output_dim])

        # -- set network ---
        feature = self.cnn.set_model(self.input, is_training=True, reuse=False)
        feature = self.conv2_networks[0].set_model(feature, self.conv2_networks[0].shortcut(feature, 256, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv2_networks)):
            feature = self.conv2_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        feature = self.conv3_networks[0].set_model(feature, self.conv3_networks[0].shortcut(feature, 512, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv3_networks)):
            feature = self.conv3_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        feature = self.conv4_networks[0].set_model(feature, self.conv4_networks[0].shortcut(feature, 1024, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv4_networks)):
            feature = self.conv4_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        feature = self.conv5_networks[0].set_model(feature, self.conv5_networks[0].shortcut(feature, 2048, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv5_networks)):
            feature = self.conv5_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        self.v_s = self.nn.set_model(feature, is_training=True, reuse=False)

        self.log_soft_max = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits = self.v_s,
            labels = self.target_val)
        self.cross_entropy = tf.reduce_mean(self.log_soft_max)

        var_list = self.cnn.get_variables()
        for conv_net in self.conv2_networks:
            var_list.extend(conv_net.get_variables())
        for conv_net in self.conv3_networks:
            var_list.extend(conv_net.get_variables())
        for conv_net in self.conv4_networks:
            var_list.extend(conv_net.get_variables())
        for conv_net in self.conv5_networks:
            var_list.extend(conv_net.get_variables())
        var_list.extend(self.nn.get_variables())
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.cross_entropy, var_list=var_list)


        # -- for test --
        feature = self.cnn.set_model(self.input, is_training=False, reuse=True)
        feature = self.conv2_networks[0].set_model(feature, self.conv2_networks[0].shortcut(feature, 256, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv2_networks)):
            feature = self.conv2_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        feature = self.conv3_networks[0].set_model(feature, self.conv3_networks[0].shortcut(feature, 512, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv3_networks)):
            feature = self.conv3_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        feature = self.conv4_networks[0].set_model(feature, self.conv4_networks[0].shortcut(feature, 1024, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv4_networks)):
            feature = self.conv4_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        feature = self.conv5_networks[0].set_model(feature, self.conv5_networks[0].shortcut(feature, 2048, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv5_networks)):
            feature = self.conv5_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        self.v_s_wo = self.nn.set_model(feature, is_training=False, reuse=True)

        self.correct_prediction = tf.equal(tf.argmax(self.v_s_wo,1), tf.argmax(self.target_val,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.output_probability = tf.nn.softmax(self.v_s_wo)


    def train(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _, loss = sess.run([self.train_op, self.cross_entropy], feed_dict = feed_dict)
        return _, loss

    def test(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _ = sess.run([self.accuracy], feed_dict = feed_dict)
        return _



class ResNet18(ResNet50):
    
    def __init__(self, fig_size, output_dim):
        super(ResNet18, self).__init__(fig_size, output_dim)
        self.cnn = _cnn('ResNetCNN', [64], [7])
        self.conv2_networks = [ResidualBlock('conv2_{}block'.format(i+1),
                                            [64, 64],
                                            [3, 3], None) for i in range(2)]

        self.conv3_networks = [ResidualBlock('conv3_{}block'.format(i+1),
                                            [128, 128],
                                            [3, 3], None) for i in range(2)]

        self.conv4_networks = [ResidualBlock('conv4_{}block'.format(i+1),
                                            [256, 256],
                                            [3, 3], None) for i in range(2)]

        self.conv5_networks = [ResidualBlock('conv5_{}block'.format(i+1),
                                            [512, 512],
                                            [3, 3], None) for i in range(2)]

        self.nn = _nn('ResNetNN', [1000, output_dim])
        self.fig_size = fig_size
        self.output_dim = output_dim
        
    def set_model(self, lr):

        self.lr = tf.Variable(
            name = "learning_rate",
            initial_value = lr,
            trainable = False)

        self.lr_op = tf.compat.v1.assign(self.lr, 0.95 * self.lr)

        # -- place holder ---
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self.fig_size, self.fig_size, 3])
        self.target_val = tf.compat.v1.placeholder(tf.float32, [None, self.output_dim])

        # -- set network ---
        feature = self.cnn.set_model(self.input, is_training=True, reuse=False)
        feature = self.conv2_networks[0].set_model(feature, self.conv2_networks[0].shortcut(feature, 64, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv2_networks)):
            feature = self.conv2_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        feature = self.conv3_networks[0].set_model(feature, self.conv3_networks[0].shortcut(feature, 128, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv3_networks)):
            feature = self.conv3_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        feature = self.conv4_networks[0].set_model(feature, self.conv4_networks[0].shortcut(feature, 256, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv4_networks)):
            feature = self.conv4_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        feature = self.conv5_networks[0].set_model(feature, self.conv5_networks[0].shortcut(feature, 512, reuse=False), is_training=True, reuse=False)
        for i in range(1, len(self.conv5_networks)):
            feature = self.conv5_networks[i].set_model(feature, feature, is_training=True, reuse=False)
        self.v_s = self.nn.set_model(feature, is_training=True, reuse=False)

        self.log_soft_max = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits = self.v_s,
            labels = self.target_val)
        self.cross_entropy = tf.reduce_mean(self.log_soft_max)

        var_list = self.cnn.get_variables()
        for conv_net in self.conv2_networks:
            var_list.extend(conv_net.get_variables())
        for conv_net in self.conv3_networks:
            var_list.extend(conv_net.get_variables())
        for conv_net in self.conv4_networks:
            var_list.extend(conv_net.get_variables())
        for conv_net in self.conv5_networks:
            var_list.extend(conv_net.get_variables())
        var_list.extend(self.nn.get_variables())
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.cross_entropy, var_list=var_list)


        # -- for test --
        feature = self.cnn.set_model(self.input, is_training=False, reuse=True)
        feature = self.conv2_networks[0].set_model(feature, self.conv2_networks[0].shortcut(feature, 64, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv2_networks)):
            feature = self.conv2_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        feature = self.conv3_networks[0].set_model(feature, self.conv3_networks[0].shortcut(feature, 128, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv3_networks)):
            feature = self.conv3_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        feature = self.conv4_networks[0].set_model(feature, self.conv4_networks[0].shortcut(feature, 256, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv4_networks)):
            feature = self.conv4_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        feature = self.conv5_networks[0].set_model(feature, self.conv5_networks[0].shortcut(feature, 512, reuse=True), is_training=False, reuse=True)
        for i in range(1, len(self.conv5_networks)):
            feature = self.conv5_networks[i].set_model(feature, feature, is_training=False, reuse=True)
        self.v_s_wo = self.nn.set_model(feature, is_training=False, reuse=True)

        self.correct_prediction = tf.equal(tf.argmax(self.v_s_wo,1), tf.argmax(self.target_val,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.output_probability = tf.nn.softmax(self.v_s_wo)