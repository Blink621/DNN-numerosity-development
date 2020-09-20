# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:01:39 2020

@author: 薛
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as fig
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def gaussian(x, *param):
    return param[0] * np.exp(-(x - param[1])**2 / (2 * param[2]**2))


# %%
namelist = ['fc1_relu', 'fc2_relu', 'fc3', 'fc3_softmax']
#namelist = ['fc3_softmax']
for name in namelist:

    numlist = np.asarray([i for i in range(1, 33)])
    actname = 'act_' + name + '.npy'
    # actname = 'act_' + name + '.npy'
    act = np.load(actname)
    infoname = 'id90_' + name + '.npy'
    info = np.load(infoname)
    info = pd.DataFrame(info)
    info.columns = ['nid', 'pn', 'r2', 'A', 'M', 'S', 'sim']
    loc = info.nid - 1
    name = name + '90new'
    data = []
    flag = 0
    for index in loc:
        index = int(index)
        nid = info.nid[flag]
        
        col1 = act[:, index, 0, 0]
        col2 = np.repeat([1,2,3], np.shape(act)[0] / 3, axis=0)
        col3 = np.tile(np.repeat(numlist, 600), 3)
        col4 = np.repeat(info.pn[flag], np.shape(act)[0], axis=0)
        col5 = np.repeat(nid, np.shape(act)[0], axis=0)
        mat = np.zeros((np.shape(act)[0], 5))
        mat[:,0] = col1
        mat[:,1] = col2
        mat[:,2] = col3
        mat[:,3] = col4
        mat[:,4] = col5
        df = pd.DataFrame(mat)
        df.columns = ['act', 'dset', 'num', 'pn', 'id']
        
        dflist = []
        for dset in range(1,4):
            dfBS = df.loc[df['dset'] == dset].copy()
            dfmean = dfBS.groupby(dfBS['num']).mean()
            dfactm = dfmean['act']
            dfmn = (dfactm - dfactm.min()) / (dfactm.max() - dfactm.min())
            dflist.append(dfactm)
        data.append(np.mean(dflist, 0))
        flag += 1
    namedata = name + '_meancurve.npy'
    np.save(namedata, data)
