# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:51:33 2020

@author: 薛
"""


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import os
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def gaussian(x, *param):
    return param[0] * np.exp(-(x - param[1])**2 / (2 * param[2]**2))


# %%
namelist = ['fc1_relu', 'fc2_relu', 'fc3']
for name in namelist:

    numlist = np.asarray([i for i in range(1, 33)])
    actname = 'act_' + name + '_test_simple.npy'
    act = np.load(actname)
    mcname = name + '80new_meancurve.npy'
    mc = np.load(mcname)
    infoname = 'id_' + name + '.npy'
    info = np.load(infoname)
    info = pd.DataFrame(info)
    info.columns = ['nid', 'pn', 'r2', 'A', 'M', 'S', 'sim']
    loc = info.nid - 1
    name = name + '80new'
    flag = 0
    simlistall = []
    for index in loc:
        index = int(index)
        nid = info.nid[flag]
        
        col1 = act[:, index, 0, 0]
        col2 = np.repeat([1,2,3,4], np.shape(act)[0] / 4, axis=0)
        col3 = np.tile(np.repeat(numlist, 600), 4)
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
        
        simlist = [info.pn[flag]]
        actarray = [mc[flag]]
        for k in range(4):
            dfone = df[df['dset'] == k+1]
            dfone = dfone.groupby(df['num']).mean()['act']
            dfonen = (dfone - dfone.min()) / (dfone.max() - dfone.min())
            actarray.append(dfonen)
        rmatrix = np.corrcoef(actarray)
        for xindex in range(1,5):
            for yindex in range(xindex):
                simlist.append(rmatrix[xindex, yindex])
        simlistall.append(simlist)
        
        flag += 1
    fileanme = 'simlist2_' + name + '.npy'
    np.save(fileanme, simlistall)
