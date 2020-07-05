# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from .variable_util import get_const_variable, get_const_weight_variable, get_rand_variable, get_dim
from .sn import spectral_norm

def linear(name, inputs, out_dim):
    in_dim = get_dim(inputs)
    w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable(name, [out_dim], 0.0)
    return tf.matmul(inputs, w) + b

def const_linear(name, inputs, out_dim, weight, bias):
    in_dim = get_dim(inputs)
    #w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    w = get_const_weight_variable(name, [in_dim, out_dim], weight)
    b = get_const_variable(name, [out_dim], bias)
    return tf.matmul(inputs, w) + b


def sn_linear(name, inputs, out_dim, update_collection):
    in_dim = get_dim(inputs)
    w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable(name, [out_dim], 0.0)
    W_shape = w.shape.as_list()
    u = tf.get_variable("u_{}".format(name), [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    return tf.matmul(inputs, spectral_norm(w, u = u, update_collection = update_collection)) + b

