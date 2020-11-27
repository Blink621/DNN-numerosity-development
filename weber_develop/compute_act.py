#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:29:55 2020

@author: zhouming
"""
import os
import torch
import numpy as np
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.core import Mask, Stimulus
from os.path import join as pjoin

# load pic to compute activation
main_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/'
model_path = pjoin(main_path, 'out/model/alexnet_train/BS_128_LR_0.001')
out_path = pjoin(main_path, 'out/activation/weber_develop')
stim_path = pjoin(main_path, 'stim/csv/num_discrim.stim.csv')
mask_path = pjoin(main_path, 'stim/csv/sig80_numerosity.dmask.csv')

stim = Stimulus()
stim.load(stim_path)
dmask = Mask()
dmask.load(mask_path)
layer_all = ['fc1_relu', 'fc2_relu']
chn = 'all'
models = os.listdir(model_path)
models.sort(key= lambda x:int(x.split('_')[1]))

#%% start computing
for model in models:
    epoch = model.split('_')[1]
    dnn = AlexNet(pretrained=False)
    dnn.model.load_state_dict(torch.load(pjoin(model_path, model))['state_dict'])
    act_all = dnn.compute_activation(stim, dmask, cuda='GPU')
    for layer in layer_all:
        act = act_all.get(layer)
        np.save(pjoin(out_path, f'act_weber_epoch{epoch}_{layer}.npy'), act)
        del act
    print(f'Finish computing epoch{epoch}!')
        
        