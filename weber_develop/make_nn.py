#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:11:48 2020

@author: zhouming
"""

import os
import numpy as np
from os.path import join as pjoin

#%% Define params
# define path
main_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/'
act_path = pjoin(main_path, 'out/activation/weber_develop/')

# define basic params
layer = 'fc1_relu'
ref = 8
epoch = 1


train_set   = np.load(pjoin(act_path, f'train/act_weber_train_data_ref{ref}_epoch{epoch}_{layer}.npy'))
train_label = np.load(pjoin(act_path, f'train/act_weber_train_label_ref{ref}_epoch{epoch}_{layer}.npy'))
test_set    = np.load(pjoin(act_path, f'test/act_weber_test_data_ref{ref}_epoch{epoch}_{layer}.npy'))
test_label  = np.load(pjoin(act_path, f'test/act_weber_test_label_ref{ref}_epoch{epoch}_{layer}.npy'))

#%% Make Linear NN
            
            
            