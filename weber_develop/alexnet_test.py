#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:27 2020

@author: zhouming
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin

main_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/'
model_path = pjoin(main_path, 'out/model/alexnet_train/BS128_LR_0.001')
epochs = np.arange(40)
top1 = np.zeros(40)
top5 = np.zeros(40)

for name in os.listdir(model_path):
    epoch = int(name.split('_')[1]) - 1
    top1[epoch] = float((name.split('_')[2]).split(':')[-1])
    top5[epoch] = float(((name.split('_')[3]).split(':')[-1]).replace('.pth',''))

#%% plot acc
    
plt.plot(epochs, top1)
plt.plot(epochs, top5)

font1 = {'family': 'serif',
        'weight': 'normal',
        'size'  : 16,}

font2 = {'family': 'serif',
        'weight': 'normal',
        'size'  : 12,}

plt.title('AlexNet training accuarcy', font1)
plt.xlabel('epoch', font2)
plt.ylabel('accuarcy:%', font2)
plt.legend(['top1', 'top5'], loc='upper left')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.show()

