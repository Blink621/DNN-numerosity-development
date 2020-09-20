#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 06:35:19 2020

@author: xuezhichao
"""

import numpy as np
act = np.load('act_fc3.npy')
for i in range(np.shape(act)[0]):
    actarray = act[i, :, 0, 0]
    expact = np.exp(actarray)
    sumexp = np.sum(expact)
    softmaxact = expact / sumexp
    act[i, :, 0, 0] = softmaxact
np.save('act_fc3_softmax.npy', act)