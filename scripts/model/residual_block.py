# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim, get_channel

class ResidualBlock(Layers):
    def __init__(self, name_scopes, layer_channels, filter_size, output_dim):
        #assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.name_scope = name_scopes
        self.layer_channels = layer_channels
        self.output_dim = output_dim
        self.filter_size = filter_size

    def set_model(self, inputs, input_shortcut, is_training = True, reuse = False):

        h  = inputs
        # convolution
        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            for i, out_chan in enumerate(self.layer_channels):
                conved = conv(inputs = h,
                    out_num = out_chan,
                    filter_width = self.filter_size[i], filter_height = self.filter_size[i],
                    stride = 1, name=i)

                bn_conved = batch_norm(i, conved, is_training)
                h = tf.nn.relu(bn_conved)

        h = tf.nn.relu(tf.add(bn_conved, input_shortcut))
        return h
    
    def shortcut(self, inputs, out_num, reuse):
        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            shortcut_input = conv(inputs = inputs,
                out_num = out_num,
                filter_width = 1, filter_height = 1,
                stride = 1, name="shortcut")
        
        return shortcut_input