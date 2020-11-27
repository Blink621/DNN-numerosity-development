#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:42:02 2020

@author: zhouming
"""
import numpy as np
import pandas as pd
from os.path import join as pjoin


main_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/stimulus' 
act_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/numerosity_unit' 
stim_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/stim/csv' 

sig = [80]
layers = ['fc1_relu', 'fc2_relu']
for isig in sig:
    with open(pjoin(stim_path, f'sig{isig}_numerosity.dmask.csv'), 'ab') as f:
        for layer in layers:
            loc = pd.read_csv(pjoin(act_path, f'id{isig}_{layer}.csv'))
            unit_id = (loc['unit_id']).values.T
            np.savetxt(f, unit_id, fmt='%d', delimiter=',', newline=',')

