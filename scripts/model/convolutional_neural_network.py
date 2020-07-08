# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim

class _network(Layers):
    def __init__(self, name_scopes, layer_channels, fc_dim, output_dim):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_channels = layer_channels
        self.fc_dim = fc_dim
        self.output_dim = output_dim

        self.name_scope_conv = u'dis_conv'
        self.name_scope_fc = u'dis_fc'

    def set_model(self, inputs, is_training = True, reuse = False):

        h  = inputs

        # convolution
        with tf.variable_scope(self.name_scope_conv, reuse = reuse):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_channels, self.layer_channels[1:])):
                if i == 0:
                    conved = conv(inputs = h,
                            out_num = out_chan,
                            filter_width = 5, filter_height = 5,
                            stride = 1, name=i)

                    h = tf.nn.relu(conved)
                else:
                    conved = conv(inputs = h,
                            out_num = out_chan,
                            filter_width = 5, filter_height = 5,
                            stride = 2, name=i)

                    bn_conved = batch_norm(i, conved, is_training)
                    h = tf.nn.relu(bn_conved)

        feature_image = h

        dim = get_dim(h)
        h = tf.reshape(h, [-1, dim])
        
        # fully connect
        with tf.variable_scope(self.name_scope_fc, reuse = reuse):
            h = linear('fc', h, self.fc_dim)
            h = batch_norm('fc', h, is_training)
            h = tf.nn.relu(h)

            h = linear('fc2', h, self.output_dim)
        return h


class ConvolutionalNeuralNetwork(object):
    
    def __init__(self, fig_size, output_dim):
        self.network = _network(["CNN"], [3,32,128,256], 512, output_dim)
        self.fig_size = fig_size
        self.output_dim = output_dim
        
    def set_model(self, lr):
        
        self.lr = tf.Variable(
            name = "learning_rate",
            initial_value = lr,
            trainable = False)

        self.lr_op = tf.assign(self.lr, 0.95 * self.lr)
        
        # -- place holder ---
        self.input = tf.placeholder(tf.float32, [None, self.fig_size, self.fig_size, 3])
        self.target_val = tf.placeholder(tf.float32, [None, self.output_dim])

        # -- set network ---
        self.v_s = self.network.set_model(self.input, is_training = True, reuse = False)

        self.log_soft_max = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.v_s,
            labels = self.target_val)
        self.cross_entropy = tf.reduce_mean(self.log_soft_max)
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)

        # -- for test --
        self.v_s_wo = self.network.set_model(self.input, is_training = False, reuse = True)
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

    def get_output(self, sess, input_data):
        feed_dict = {self.input: input_data}
        _ = sess.run([self.output_probability], feed_dict = feed_dict)
        return _