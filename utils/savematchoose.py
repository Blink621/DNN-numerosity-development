#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:35:58 2020

@author: xuezhichao
"""

import scipy.io
import numpy as np
data1 = np.load('act_fc1_relu.npy')
data2 = np.load('act_fc2_relu.npy')
data3 = np.load('act_fc3.npy')
data4 = np.load('act_fc3_softmax.npy')

scipy.io.savemat('act_choose.mat', {'fc1_relu' : data1, 'fc2_relu' : data2, 'fc3' : data3, 'dc3_softmax' : data4})