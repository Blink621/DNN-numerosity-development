# This file is the old version, used to compute activation
# but the order in act.npy is wrong

import os
import matplotlib.pyplot as plt
import numpy as np
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.core import Mask
from os.path import join as pjoin
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score

# load pic to compute activation
main_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/stimulus' 
out_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/activation' 
stim_set = ['standard-circle', 'density-circle', 'pentagon-mixed']
dnn = AlexNet()
layer_all = ['fc1_relu', 'fc2_relu']
chn = 'all'
# start computing
for stim in stim_set:
    stim_path = pjoin(main_path, stim)
    pic_all = np.zeros((len(os.listdir(stim_path)), 3, 224, 224), dtype=np.uint8)
    for idx,pic_name in enumerate(os.listdir(stim_path)):
        pic_path = pjoin(stim_path, pic_name)
        pic = plt.imread(pic_path).astype(np.uint8)
        pic_all[idx] = pic.transpose((2,0,1))
    for layer in layer_all:
        mask = Mask(layer,chn)
        act = dnn.compute_activation(pic_all, mask).get(layer)
        np.save(pjoin(out_path, f'{stim}_{layer}_act.npy'), act)

# using activation to select the numerosity unit
    
    
#out_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/activation' 
#txt_file = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/sig_info.txt' 
#layer_all = ['fc1_relu', 'fc2_relu']
#
#for layer in layer_all:
#    numer = 0
#    act_standard = np.load(pjoin(out_path, f'standard-circle_{layer}_act.npy'))
#    act_density = np.load(pjoin(out_path, f'density-circle_{layer}_act.npy'))
#    act_pentagon = np.load(pjoin(out_path, f'pentagon-mixed_{layer}_act.npy'))
#    # loop
#    unit_num = act_standard.shape[1]
#    unit_loc = np.zeros((unit_num, len(layer_all)))
#    for idx in range(unit_num):
#        get = lambda x : x[:, idx]
#        col1 = np.vstack((get(act_standard), get(act_density), get(act_pentagon))).flatten()
#        col2 = np.tile(np.repeat(range(1, 33), 600),3)
#        col3 = np.repeat([1,2,3], 32*600, axis=0)
#        mat = np.zeros((32*600*3, 3))
#        mat[:,0] = col1
#        mat[:,1] = col2
#        mat[:,2] = col3
#        df = pd.DataFrame(mat)
#        df.columns = ['act', 'num', 'dset']
#        formula = 'act~C(num)+C(dset)+C(num):C(dset)'
#        res = anova_lm(ols(formula, df).fit())
#        sig = res['PR(>F)']
##        sig_info = 'num:%.2f, set:%.2f, interact:%.2f'%(sig['C(num)'], sig['C(dset)'], sig['C(num):C(dset)'])
##        print(sig_info)
#        if (sig['C(dset)'] > 0.05):# & (sig['C(num)'] < 0.05) & (sig['C(num):C(dset)'] > 0.05):
#            if layer == 'fc1':
#                unit_loc[idx, 0] = 1 
#            else:
#                unit_loc[idx, 1] = 1 
#            numer += 1
#            print(f'Find {numer} numerosity units in {layer} already')
#        #print(f'Finish computing units: {idx+1}/{unit_num} in {layer}')
#
#np.save(pjoin(out_path, f'loc.npy'), unit_loc)
#
#def gaussian(x, *param):
#    return param[0] * np.exp(-(x - param[1])**2 / (2 * param[2]**2))
#
#f = open('sig_info.txt', 'a')
#
#numlist = np.asarray(range(1,33))
#for layer in layer_all:
#    act_standard = np.load(pjoin(out_path, f'standard-circle_{layer}_act.npy'))
#    act_density = np.load(pjoin(out_path, f'density-circle_{layer}_act.npy'))
#    act_pentagon = np.load(pjoin(out_path, f'pentagon-mixed_{layer}_act.npy'))
#    # loop
#    unit_num = act_standard.shape[1]
##    sig_info = np.zeros((unit_num, 2))
#    for idx in range(unit_num):
#        get = lambda x : x[:, idx]
#        col1 = np.vstack((get(act_standard), get(act_density), get(act_pentagon))).flatten()
#        col2 = np.tile(np.repeat(numlist, 600),3)
#        col3 = np.repeat([1,2,3], 32*600, axis=0)
#        mat = np.zeros((32*600*3, 3))
#        mat[:,0] = col1
#        mat[:,1] = col2
#        mat[:,2] = col3
#        df = pd.DataFrame(mat)
#        df.columns = ['act', 'num', 'set']
#        if 0 not in df['act'].values:
#            df_mean = df.groupby([df['set'], df['num']]).mean()['act']
#            norm = lambda x : (x - x.min()) / (x.max() - x.min())
#            # compute correlation
#            x_set = []
#            for iset in [1,2,3]:
#                df_set = norm(df_mean[iset])
#                x_set.append(df_set)
#            corr = (np.corrcoef(x_set).sum() - 3) / 6
#            #compute r socre
#            df_r = df.groupby([df['num']]).mean()['act']
#            popt, pcov = curve_fit(gaussian, np.log2(numlist), norm(df_r), p0=[5, 5, 5], bounds=([0, 0, 0], [np.inf, np.log2(33), np.inf]), maxfev=5000000)
#            r2 = r2_score(norm(df_r), gaussian(np.log2(numlist), *popt))
#            info = 'cor:%.2f; r_score:%.2f\n'%(corr, r2)
##            sig_info[idx, 0] = corr
##            sig_info[idx, 1] = r2
#        else:
#            info = 'Act contain 0\n'
#        f.write(info)
#
#f.close()
