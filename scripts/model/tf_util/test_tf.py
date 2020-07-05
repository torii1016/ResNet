# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pickle

from tf_sac import AggActor

if __name__ == '__main__':
    dir_name = '/home/takatomo/tests/hoge/pegin_chainer/scripts/pickled_weight'
    agged = ['yoko_naga', 'tate_naga']

    agged_files = [os.path.join(dir_name, '{}.pickle'.format(_)) for _ in agged]

    actor = AggActor(6, 'ActorNetowrk', agged_files)

    state_ph = tf.placeholder(dtype = tf.float32, shape = [None, 12])

    actor.set_model(state_ph, True, False)
    
    
