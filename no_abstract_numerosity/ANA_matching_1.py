# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:45:33 2020

@author: 薛
"""


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# %%
com_ratio = np.asarray([0.4, 0.7, 1, 1.3, 1.6])


#%%
classres = np.load('match_classres_fc1_80dset0.npy')
labels = np.load('match_labels_fc1_80dset0.npy')
samdata = np.load('match_samdata_fc1_80dset0.npy')

sammean = samdata[:, 3:]
sammeanmin = np.expand_dims(sammean.min(axis=1), axis=1)
sammeanmax = np.expand_dims(sammean.max(axis=1), axis=1)
sammean = (sammean - sammeanmin) / (sammeanmax - sammeanmin)
sammean[np.isnan(sammean)] = 0
sammean = np.mean(samdata, axis=1)
sammean = np.expand_dims(sammean, axis=1)
sammeandata = np.concatenate((samdata[:, :3], sammean), axis=1)

current_index = (classres == labels)
current_index = np.expand_dims(current_index, axis=1)


# %%
num_distance_list = []
# num_sam_list = []
# num_com_list = []
for sam in range(1, 33):
    com_list = np.round(com_ratio * sam)
    com_list = com_list[(com_list<=32) & (com_list>=1)]
    com_list = np.unique(com_list)
    for com in com_list:
        num_distance = sam - com
        num_distance_list.append(num_distance)
        # num_sam_list.append(sam)
        # num_com_list.append(com)
num_distance_list = np.expand_dims(np.repeat(num_distance_list, 100), axis=1)
# num_sam_list = np.expand_dims(np.repeat(num_sam_list, 100), axis=1)
# num_com_list = np.expand_dims(np.repeat(num_com_list, 100), axis=1)

# df = np.concatenate((current_index, num_sam_list, num_com_list, num_distance_list, sammeandata), axis=1)
df = np.concatenate((current_index, num_distance_list, sammeandata), axis=1)
# df = np.concatenate((current_index, num_distance_list, sammean), axis=1)
df = pd.DataFrame(df)
# df.columns = ['test_acc', 'num_sam', 'num_com', 'num_dis', 'num_set', 'num_pre', 'num_index', 'act_mean']
df.columns = ['test_acc', 'num_dis', 'num_set', 'num_pre', 'num_index', 'act_mean']

dferror = df.groupby(df['test_acc']).get_group(0)[['num_pre', 'num_dis', 'act_mean']].copy()
dferrorG = dferror.groupby(dferror['num_dis'])
dferrorMall = dferrorG.mean()['act_mean']
dferrorMall = np.squeeze(np.asarray(dferrorMall))
dferrorSEall = dferrorG.std() / np.sqrt(dferrorG.count())
dferrorSEall = np.squeeze(np.asarray(dferrorSEall['act_mean']))
num_dis_list = np.unique(dferror['num_dis'])
plt.figure(figsize=(20, 15))
plt.errorbar(num_dis_list, dferrorMall, yerr=dferrorSEall, fmt='k', ecolor='black', elinewidth=2)

dfcorrent = df.groupby(df['test_acc']).get_group(1)[['num_pre', 'num_dis', 'act_mean']].copy()
dfcorrentG = dfcorrent.groupby(dfcorrent['num_dis'])
dfcorrentMall = dfcorrentG.mean()['act_mean']
dfcorrentMall = np.squeeze(np.asarray(dfcorrentMall))
dfcorrentSEall = dfcorrentG.std() / np.sqrt(dferrorG.count())
dfcorrentSEall = np.squeeze(np.asarray(dfcorrentSEall['act_mean']))
num_dis_list = np.unique(dfcorrent['num_dis'])
plt.figure(figsize=(20, 15))
plt.errorbar(num_dis_list, dfcorrentMall, yerr=dfcorrentSEall, fmt='k', ecolor='black', elinewidth=2)

plt.figure(figsize=(20, 15))
num_dis_list = np.unique(dferror['num_dis'])
plt.errorbar(num_dis_list, dferrorMall, yerr=dferrorSEall, fmt='r', ecolor='red', elinewidth=5)
num_dis_list = np.unique(dfcorrent['num_dis'])
plt.errorbar(num_dis_list, dfcorrentMall, yerr=dfcorrentSEall, fmt='g', ecolor='green', elinewidth=2)
