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
from os.path import join as pjoin
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def gaussian(x, *param):
    return param[0] * np.exp(-(x - param[1])**2 / (2 * param[2]**2))

out_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/images/no_numer' 
act_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/activation' 
num_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/numerosity_unit' 

# %%
namelist = ['fc1_relu', 'fc2_relu', 'fc3']
for name in namelist:

    numlist = np.asarray([i for i in range(1, 33)])
    actname = 'act_' + name + '_test_simple.npy'  # define the act file name
    act = np.load(pjoin(act_path, actname))  # load the act data of neure
    infoname = 'id80_' + name + '.npy'  # define the information file name
    info = np.load(pjoin(num_path, infoname))  # load the information data of neure
    info = pd.DataFrame(info)  # turns to a DataFrame
    info.columns = ['nid', 'pn', 'r2', 'A', 'M', 'S', 'sim']
    loc = info.nid - 1
    name = name + '90new'
    flag = 0
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
        if flag == 0:
            dfall = df
        else:
            dfall = pd.concat([dfall, df], ignore_index=True)
        flag += 1
    
    
    # %%
    plt.figure(figsize=(20, 15))
    for i in range(8):
        for j in range(4):
            PN = numlist[i * 4 + j]
            df = dfall[dfall['pn'] == PN].copy()
            dfm = df.groupby(df['num']).mean()['act']
            dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
    
            scale = dfm.max() - dfm.min()
            dfstd = df.groupby(df['num']).std()['act']
            dferr = dfstd / ((df.count() / 32)['act'] ** 0.5)
            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}',fontsize=10)
            
            if len(df) != 0:
                for k in range(4):
                    dfm1 = df[df['dset'] == k+1]
                    dfm1 = dfm1.groupby(df['num']).mean()['act']
                    dfm1n = (dfm1 - dfm1.min()) / (dfm1.max() - dfm1.min())
                    ax.plot(dfm1, linewidth=0.7)
                if PN == 1:
                    ax.legend(fontsize=10)
                plt.errorbar(numlist, dfm, yerr=dferr, fmt='k', ecolor='black', elinewidth=2)
    picname = 'curve_ro_' + name + '.png'
    plt.savefig(pjoin(out_path, picname))
    plt.close('all')
    print('Finish one figure')
    
    # %%
    plt.figure(figsize=(20, 15))
    for i in range(8):
        for j in range(4):
            PN = numlist[i * 4 + j]
            df = dfall[dfall['pn'] == PN].copy()
            dfm = df.groupby(df['num']).mean()['act']
            dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
    
            scale = dfm.max() - dfm.min()
            dfstd = df.groupby(df['num']).std()['act'] / scale
            dferr = dfstd / ((df.count() / 32)['act'] ** 0.5)
            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}',fontsize=10)
            
            dfrec = []
            if len(df) != 0:
                for k in range(4):
                    dfm1 = df[df['dset'] == k+1]
                    dfm1 = dfm1.groupby(df['num']).mean()['act']
                    dfm1n = (dfm1 - dfm1.min()) / (dfm1.max() - dfm1.min())
                    ax.plot(dfm1n, linewidth=0.7)
                    dfrec.append(dfm1n)
                if PN == 1:
                    ax.legend(fontsize=10)
                plt.plot(numlist, np.mean(dfrec, 0), 'k', linewidth=2)
    picname = 'curve_no_' + name + '.png'
    plt.savefig(pjoin(out_path, picname))
    plt.close('all')
    print('Finish one figure')
    
        # %%
    plt.figure(figsize=(20, 15))
    for i in range(8):
        for j in range(4):
            PN = numlist[i * 4 + j]
            df = dfall[dfall['pn'] == PN].copy()
            dfm = df.groupby(df['num']).mean()['act']
            dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
    
            scale = dfm.max() - dfm.min()
            dfstd = df.groupby(df['num']).std()['act']
            dferr = dfstd / ((df.count() / 32)['act'] ** 0.5)
            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}',fontsize=10)
            
            if len(df) != 0:
                for k in range(4):
                    dfm1 = df[df['dset'] == k+1]
                    dfm1 = dfm1.groupby(df['num']).mean()['act']
                    dfm1n = (dfm1 - dfm1.min()) / (dfm1.max() - dfm1.min())
                    ax.plot(dfm1, linewidth=0.7)
                if PN == 1:
                    ax.legend(fontsize=10)
    picname = 'curve_ro_nm_' + name + '.png'
    plt.savefig(pjoin(out_path, picname))
    plt.close('all')
    print('Finish one figure')
    
    # %%
    plt.figure(figsize=(20, 15))
    for i in range(8):
        for j in range(4):
            PN = numlist[i * 4 + j]
            df = dfall[dfall['pn'] == PN].copy()
            dfm = df.groupby(df['num']).mean()['act']
            dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
    
            scale = dfm.max() - dfm.min()
            dfstd = df.groupby(df['num']).std()['act'] / scale
            dferr = dfstd / ((df.count() / 32)['act'] ** 0.5)
            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}',fontsize=10)
            
            if len(df) != 0:
                for k in range(4):
                    dfm1 = df[df['dset'] == k+1]
                    dfm1 = dfm1.groupby(df['num']).mean()['act']
                    dfm1n = (dfm1 - dfm1.min()) / (dfm1.max() - dfm1.min())
                    ax.plot(dfm1n, linewidth=0.7)
                if PN == 1:
                    ax.legend(fontsize=10)
    picname = 'curve_no_nm_' + name + '.png'
    plt.savefig(pjoin(out_path, picname))
    plt.close('all')
    print('Finish one figure')


    # %%
#    dfall_back = dfall.copy()
#    for num_of_set in range(1,5):
#        
#        dfall = dfall_back[dfall_back['dset'] == num_of_set]
#        
#        sigmaline = []
#        sigmalog = []
#        sigmapow2 = []
#        sigmapow3 = []
#        r2line = []
#        r2log = []
#        r2pow2 = []
#        r2pow3 = []
#        plt.figure(figsize=(20, 15))
#        for i in range(8):
#            for j in range(4):
#                PN = numlist[i * 4 + j]
#                df = dfall[dfall['pn'] == PN].copy()
#                dfm = df.groupby(df['num']).mean()['act']
#                dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
#        
#                ax = plt.subplot(8, 4, (i)*4+j+1)
#                ax.set_xticks([])
#                ax.set_title(f'PN = {PN}', fontsize=10)
#                plt.ylim(-0.1, 1.2)
#    
#                ax.plot(dfmn)
#                if len(dfm) != 0:
#                    popt, pcov = curve_fit(gaussian, numlist, dfmn, p0=[15, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
#                    plt.plot(numlist, gaussian(numlist, *popt))
#                    sigmaline.append(popt[2])
#                    r2 = r2_score(dfmn, gaussian(numlist, *popt))
#                    r2line.append(r2)
#            
#                    popt, pcov = curve_fit(gaussian, np.log2(numlist), dfmn, p0=[15, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
#                    sigmalog.append(popt[2])
#                    r2 = r2_score(dfmn, gaussian(np.log2(numlist), *popt))
#                    r2log.append(r2)
#            
#                    popt, pcov = curve_fit(gaussian, numlist**(1/2), dfmn, p0=[15, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
#                    sigmapow2.append(popt[2])
#                    r2 = r2_score(dfmn, gaussian(numlist**(1/2), *popt))
#                    r2pow2.append(r2)
#            
#                    popt, pcov = curve_fit(gaussian, numlist**(1/3), dfmn, p0=[15, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
#                    sigmapow3.append(popt[2])
#                    r2 = r2_score(dfmn, gaussian(numlist**(1/3), *popt))
#                    r2pow3.append(r2)
#        picname = f'{num_of_set}_gaussian_' + name + '.png'
#        plt.savefig(picname)
#        plt.close('all')
#    
#    
#        # %%
#    
#        plt.figure(figsize=(20, 15))
#        ax = plt.subplot(111)
#        plt.tick_params(labelsize=20)
#        num = dfall.groupby(dfall['pn']).mean().index
#        ax.set_xticks(num)
#        plt.plot(num, np.asarray(np.abs(sigmaline)) / (2**0.5), label = 'line')
#        plt.plot(num, np.asarray(np.abs(sigmalog)) / (2**0.5), label = 'log')
#        plt.plot(num, np.asarray(np.abs(sigmapow2)) / (2**0.5), label = 'pow1/2')
#        plt.plot(num, np.asarray(np.abs(sigmapow3)) / (2**0.5), label = 'pow1/3')
#        plt.xlabel('PN', fontsize=30)
#        plt.ylabel('sigma', fontsize=30)
#        ax.legend(fontsize=30)
#        picname = f'{num_of_set}_sigma_' + name + '.png'
#        plt.savefig(picname)
#        plt.close('all')
#    
#    
#        # %%
#        length = len(num)
#        plt.figure(figsize=(20, 15))
#        mat = np.zeros((length * 4, 2))
#        mat[:,0] = r2line + r2log + r2pow2 + r2pow3
#        mat[:,1] = ['1'] * length + ['2'] * length + ['3'] * length + ['4'] * length
#        dfr2 = pd.DataFrame(mat)
#        dfr2.columns = ['r2', 'scale']
#        formula = 'r2~C(scale)'
#        res = anova_lm(ols(formula, dfr2).fit())
#        print(res);
#    
#        r2 = dfr2.groupby(dfr2['scale']).mean()['r2']
#        r2se = dfr2.groupby(dfr2['scale']).std()['r2'] / length**0.5
#    
#        plt.ylim(0.6, 1)
#        plt.tick_params(labelsize=30)
#        plt.xlabel('Scale', fontsize=40)
#        plt.ylabel('Goodness of fit (r2)', fontsize=40)
#        plt.bar(['line', 'log', 'pow1/2', 'pow1/3'], r2, yerr=r2se, error_kw={'capsize' : 6})
#    
#        res1 = stats.ttest_rel(dfr2[dfr2['scale']==1]['r2'], dfr2[dfr2['scale']==2]['r2'])
#        res2 = stats.ttest_rel(dfr2[dfr2['scale']==1]['r2'], dfr2[dfr2['scale']==3]['r2'])
#        res3 = stats.ttest_rel(dfr2[dfr2['scale']==1]['r2'], dfr2[dfr2['scale']==4]['r2'])
#        print('log', res1[1], 'pow1/2', res2[1], 'pow1/3', res3[1])
#        picname = f'{num_of_set}_r2_' + name + '.png'
#        plt.savefig(picname)
#        plt.close('all')
