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
namelist = ['fc1_relu', 'fc2_relu', 'fc3']
for name in namelist:

    numlist = np.asarray([i for i in range(1, 33)])
    actname = 'act_' + name + '_test_simple.npy'
    # actname = 'act_' + name + '.npy'
    act = np.load(actname)
    infoname = 'id_' + name + '.npy'
    info = np.load(infoname)
    info = pd.DataFrame(info)
    info.columns = ['nid', 'pn', 'r2', 'A', 'M', 'S', 'sim']
    loc = info.nid - 1
    name = name + '80new'
    data = []
    flag = 0
    for index in loc:
        index = int(index)
        
        col1 = act[:, index, 0, 0]
        col2 = np.repeat([1,2], np.shape(act)[0] / 2, axis=0)
        col3 = np.tile(np.repeat([1,2], np.shape(act)[0] / 4, axis=0), 2)
        col4 = np.tile(np.repeat(numlist, 600), 4)
        col5 = np.repeat(info.pn[flag], np.shape(act)[0], axis=0)
        col6 = np.repeat(info.nid[flag], np.shape(act)[0], axis=0)
        mat = np.zeros((np.shape(act)[0], 6))
        mat[:,0] = col1
        mat[:,1] = col2
        mat[:,2] = col3
        mat[:,3] = col4
        mat[:,4] = col5
        mat[:,5] = col6
        df = pd.DataFrame(mat)
        df.columns = ['act', 'color', 'dset', 'num', 'pn', 'id']
        
        datalist = [info.nid[flag]]
        pnlist = [info.pn[flag]]
        centerlist = [info.M[flag]]
        r2list = [info.r2[flag]]
        for color in range(1,3):
            dfB = df.loc[df['color'] == color].copy()
            for dset in range(1,3):
                dfBS = dfB.loc[dfB['dset'] == dset].copy()
                dfmean = dfBS.groupby(dfBS['num']).mean()
                dfactm = dfmean['act']
                maxdfmean = np.max(dfmean['act'])
                if maxdfmean == 0:
                    dpn = 0
                else:
                    dpn = dfmean[dfmean.loc[:,'act'].isin([maxdfmean])].index[0]
                pnlist.append(dpn)
                
                popt, pcov = curve_fit(gaussian, np.log2(numlist), dfactm, p0=[3, 3, 3], bounds=([0, 0, 0], [np.inf, np.log2(33), np.inf]), maxfev=5000000)
                r2 = r2_score(dfactm, gaussian(np.log2(numlist), *popt))
                centerlist.append(popt[1])
                r2list.append(r2)
        datalist = datalist + pnlist + centerlist + r2list
        data.append(datalist)
        flag += 1
    namedata = name + '_ANOVA.npy'
    np.save(namedata, data)
