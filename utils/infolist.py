# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:48:36 2020

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


filelist = ['act_fc1_relu.npy', 'act_fc2_relu.npy', 'act_fc3.npy', 'act_fc3_softmax.npy']



# numlist = [1] + [2*i for i in range(1, 16)]
# numlist = np.asarray(numlist)
# colorlist = ['red', 'blue', 'green']
# labellist = ['dset1', 'dset2', 'dset3']

numlist = np.asarray(range(1,33))
numlist = np.asarray(numlist)
for filename in filelist:
    act = np.load(filename)
    exloc = []
    for fmap in range(np.shape(act)[1]):
        for i in range(np.shape(act)[2]):
            for j in range(np.shape(act)[3]):
                exloc.append([fmap, i, j])

    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    
    infolist = []
    nid = 0
    for locxyz in exloc:
        nid += 1
        col1 = act[:, locxyz[0], locxyz[1], locxyz[2]]
        col2 = np.repeat([1,2,3], np.shape(act)[0] / 3, axis=0)
        col3 = np.tile(np.repeat(numlist, 600),3)
        col4 = np.zeros(np.shape(act)[0])
        col5 = np.repeat(nid, np.shape(act)[0], axis=0)
        mat = np.zeros((np.shape(act)[0], 5))
        mat[:,0] = col1
        mat[:,1] = col2
        mat[:,2] = col3
        mat[:,3] = col4
        mat[:,4] = col5
        df = pd.DataFrame(mat)
        df.columns = ['act', 'dset', 'num', 'pn', 'id']
    
        dfmean = df.groupby(df['num']).mean()
        dfactm = dfmean['act']
        maxdfmean = np.max(dfmean['act'])
        if maxdfmean == 0:
            pn = 0
        else:
            pn = dfmean[dfmean.loc[:,'act'].isin([maxdfmean])].index[0]
        df['pn'] = np.repeat(pn, np.shape(act)[0])
# %%
        judge = np.min(df['act'].groupby(df['dset']).max())
        if judge != 0:
            dfm = df.groupby(df['num']).mean()['act']
            dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
            try:
                popt, pcov = curve_fit(gaussian, np.log2(numlist), dfmn, p0=[15, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
                r2 = r2_score(dfmn, gaussian(np.log2(numlist), *popt))
                actnum = []
                actnumn = []
                for index in range(3):
                    dfnum = df[df['dset'] == index+1]
                    dfnum = dfnum.groupby(dfnum['num']).mean()['act']
                    dfnumn = (dfnum - dfnum.min()) / (dfnum.max() - dfnum.min())
                    actnum.append(dfnum)
                    actnumn.append(dfnumn)
                sim = (np.corrcoef(actnum).sum() - 3) / 6
                infolist.append([nid, df['pn'].max(), r2, popt[0], popt[1], popt[2], sim])
            except:
                print(df['pn'].max())
    idfile = 'info_' + filename[4:-4] + '.npy'
    np.save(idfile, infolist)

