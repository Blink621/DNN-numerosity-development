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
#define path
main_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/'
act_path = pjoin(main_path, 'out/activation/weber_develop/raw')
out_path = pjoin(main_path, 'out/activation/weber_develop/')
#define other params
train_size = 500
validate_size = 50
test_size = 50
epoch = 1
layer = 'fc1_relu'
#layer_all = ['fc1_relu', 'fc2_relu']

#for epoch in range(40):
#    for layer in layer_all:
#        act_raw = np.load(f'act_weber_epoch{epoch+1}_{layer}.npy')
act_raw = np.load(pjoin(act_path, f'act_weber_epoch{epoch}_{layer}.npy')).squeeze()
act_size = act_raw[:int(act_raw.shape[0]/2), :] #size: 15000
act_area = act_raw[int(act_raw.shape[0]/2):, :]  


#%% making sets
# make act array. ref means reference number
for ref in [8, 16]:
    if ref == 8:
        n2_of_ref = np.delete(np.arange(4,15), 4)# reference num:8, the compare num is 4:14 except 8
    else:
        n2_of_ref = np.delete(np.arange(8,29), 8)# reference num:16, the compare num is 8:28 except 16     
    # define set and labels
    # ref8: train:10000 test:1000 ; ref16: train:20000 test:2000
    train_single = n2_of_ref.shape[0]*500
    test_single  = n2_of_ref.shape[0]*50
    train_set   = np.zeros((train_single*2, act_raw.shape[1])) 
    test_set    = np.zeros((test_single*2, act_raw.shape[1]))
    train_label = np.zeros((train_single*2, 2))
    test_label  = np.zeros((test_single*2, 2))
    # load act into train and test set
    loop = 0
    for dataset in ['size', 'area']:
        
        # define act in size and area
        if dataset == 'size':
            act_ref = act_size[(600*(ref-1)):600*ref, :] #size: 600
            act_com = act_size
        else:
            act_ref = act_area[(600*(ref-1)):600*ref, :] #size: 600
            act_com = act_area
        # get train and test
        act_ref_train = act_ref[:train_size, :] #size: 500
        act_ref_test  = act_ref[600-test_size:, :]  #size: 50
        # extract act for compare numbers
        for (idx_n2, n2) in enumerate(n2_of_ref):
            actual_n2 = n2-4 # Using actual n2 as the dataset num is 4:28. The absolute number is 25.
            start_idx = 600*actual_n2
            end_idx   = 600*(actual_n2+1)
            # define compare num act in each n2
            act_com_train = act_com[start_idx:start_idx+train_size, :]
            act_com_test  = act_com[end_idx-test_size:end_idx, :]            
            # shuffle sequence to prevent order effect. This means the order of within compare num.
            order_train = np.random.permutation(np.hstack((np.ones(250), np.zeros(250))))
            order_test  = np.random.permutation(np.hstack((np.ones(25),  np.zeros(25))))
            # loop in train set
            for (idx_order, ilabel) in enumerate(order_train):
                # definition of ilabel: 1 ref>com; 0 com>ref
                # integrate two vector: subtract/joint. Here now is subtract
                if ilabel == 1:
                    stim = act_ref_train[idx_order, :] - act_com_train[idx_order, :]
                    label = [1, 0]
                else:
                    stim = act_com_train[idx_order, :] - act_ref_train[idx_order, :] 
                    label = [0, 1]
                # merge stim and label into train set
                train_set[loop*train_single + idx_n2*train_size + idx_order, :] = stim
                train_label[loop*train_single + idx_n2*train_size + idx_order, :] = label
            # loop in test set
            for (idx_order, ilabel) in enumerate(order_test):
                if ilabel == 1:
                    stim = act_ref_test[idx_order, :] - act_com_test[idx_order, :]
                    label = [1, 0]
                else:
                    stim = act_com_test[idx_order, :] - act_ref_test[idx_order, :] 
                    label = [0, 1]
                # merge stim and label into test set
                test_set[loop*test_single + idx_n2*test_size + idx_order, :] = stim
                test_label[loop*test_single + idx_n2*test_size + idx_order, :] = label
            # shuffle all num order in train set. This means the order of between compare num.
            # The data and label using the same shuffle order
            state = np.random.get_state()
            np.random.shuffle(train_set)
            np.random.set_state(state)
            np.random.shuffle(train_label)
        loop += 1
    #%% save data  
    np.save(pjoin(out_path, f'train/act_weber_train_data_ref{ref}_epoch{epoch}_{layer}.npy'), train_set)
    np.save(pjoin(out_path, f'train/act_weber_train_label_ref{ref}_epoch{epoch}_{layer}.npy'), train_label)
    np.save(pjoin(out_path, f'test/act_weber_test_data_ref{ref}_epoch{epoch}_{layer}.npy'), test_set)
    np.save(pjoin(out_path, f'test/act_weber_test_label_ref{ref}_epoch{epoch}_{layer}.npy'), test_label)
    print(f'Finish making ref{ref} dataset in epoch{epoch}!')        
            
            
            