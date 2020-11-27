#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:33:24 2020

@author: xuezhichao
"""

import numpy as np
from PIL import Image
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.core import Mask
from os.path import join as pjoin
import matplotlib.pyplot as plt
import os
from dnnbrain.dnn.core import Stimulus

dnn = AlexNet()

#layer = 'fc1_relu'
#chn = 'all'
#mask = Mask(layer, chn)
#stim = Stimulus()
#stim.load('sci_test_simple.stim.csv')
#act_fc1_relu = dnn.compute_activation(stim, mask).get(layer)
#np.save('act_fc1_relu_test_simple.npy', act_fc1_relu)
#
#layer = 'fc2_relu'
#chn = 'all'
#mask = Mask(layer, chn)
#stim = Stimulus()
#stim.load('sci_test_simple.stim.csv')
#act_fc2_relu = dnn.compute_activation(stim, mask).get(layer)
#np.save('act_fc2_relu_test_simple.npy', act_fc2_relu)

layer = 'fc3'
chn = 'all'
mask = Mask(layer, chn)
stim = Stimulus()
stim.load('sci_test_simple.stim.csv')
act_fc3 = dnn.compute_activation(stim, mask).get(layer)
np.save('act_fc3_test_simple.npy', act_fc3)

layer = 'fc1_relu'
chn = 'all'
mask = Mask(layer, chn)
stim = Stimulus()
stim.load('sci_test.stim.csv')
act_fc1_relu = dnn.compute_activation(stim, mask).get(layer)
np.save('act_fc1_relu_test.npy', act_fc1_relu)

layer = 'fc2_relu'
chn = 'all'
mask = Mask(layer, chn)
stim = Stimulus()
stim.load('sci_test.stim.csv')
act_fc2_relu = dnn.compute_activation(stim, mask).get(layer)
np.save('act_fc2_relu_test.npy', act_fc2_relu)

layer = 'fc3'
chn = 'all'
mask = Mask(layer, chn)
stim = Stimulus()
stim.load('sci_test.stim.csv')
act_fc3 = dnn.compute_activation(stim, mask).get(layer)
np.save('act_fc3_test.npy', act_fc3)