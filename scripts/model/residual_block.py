# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim

class ResidualBlock(Layers):
    def __init__(self, name_scopes, layer_channels, filter_size, output_dim):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.name_scope = name_scopes
        self.layer_channels = layer_channels
        self.output_dim = output_dim
        self.filter_size = filter_size

    def set_model(self, inputs, is_training = True, reuse = False):

        h  = inputs
        # convolution
        with tf.variable_scope(self.name_scope, reuse = reuse):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_channels, self.layer_channels[1:])):
                conved = conv(inputs = h,
                    out_num = out_chan,
                    filter_width = self.filter_size[i], filter_height = self.filter_size[i],
                    stride = 1, name=i)

                bn_conved = batch_norm(i, conved, is_training)
                h = tf.nn.relu(bn_conved)

            shortcut_input = conv(inputs = inputs,
                out_num = self.layer_channels[-1],
                filter_width = 1, filter_height = 1,
                stride = 1, name="shortcut")

        h = tf.nn.relu(tf.add(bn_conved, shortcut_input))
        return h