# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:45:33 2020

@author: 薛
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
com_ratio = np.asarray([0.4, 0.7, 1, 1.3, 1.6])


# %%
classres = np.load('match_classres_fc1_80dset0.npy')
labels = np.load('match_labels_fc1_80dset0.npy')
samdata = np.load('match_samdata_fc1_80dset0.npy')
id_prenum = np.load('id80_fc1_relu.npy')

sam = samdata[:, 3:]
sammin = np.expand_dims(sam.min(axis=1), axis=1)
sammax = np.expand_dims(sam.max(axis=1), axis=1)
sam = (sam - sammin) / (sammax - sammin)
samdata = np.concatenate((samdata[:, :3], sam), axis=1)

prenum_index = id_prenum[:, 1]
current_index = (classres == labels)
current_index = np.expand_dims(current_index, axis=1)


# %%
num_distance_list = []
num_sam_list = []
num_com_list = []
for sam in range(1, 33):
    com_list = np.round(com_ratio * sam)
    com_list = com_list[(com_list<=32) & (com_list>=1)]
    com_list = np.unique(com_list)
    for com in com_list:
        num_distance = sam - com
        num_distance_list.append(num_distance)
        num_sam_list.append(sam)
        num_com_list.append(com)
num_distance_list = np.expand_dims(np.repeat(num_distance_list, 100), axis=1)
num_sam_list = np.expand_dims(np.repeat(num_sam_list, 100), axis=1)
num_com_list = np.expand_dims(np.repeat(num_com_list, 100), axis=1)

# data_all = np.concatenate((current_index, num_sam_list, num_com_list, num_distance_list, samdata), axis=1)
data_all = np.concatenate((current_index, num_sam_list, num_com_list, num_distance_list, samdata), axis=1)
data = data_all[data_all[:, 0] == False, :]
# data = data_all[data_all[:, 0] == True, :]

# %%
num_dis_list = []
SUM_list = []
M_list = []
STD_list= []
LEN_list = []

for sam in range(1, 33):
    prenum_choose = (prenum_index == sam)
    data_sam = data[data[:, 1] == sam]
    act_data = data_sam[:, 7:]
    act_data = act_data[:, prenum_choose]
    data_sam = np.concatenate((data_sam[:, :7], act_data), axis=1)
    
    SUM_list_one = []
    M_list_one = []
    STD_list_one= []
    LEN_list_one = []
    
    com_list = np.round(com_ratio * sam)
    com_list = com_list[(com_list<=32) & (com_list>=1)]
    com_list = np.unique(com_list)
    num_dis_list_one = sam - com_list
    
    for num_dis in num_dis_list_one:
        data_sam_numdis = data_sam[data_sam[:, 3] == num_dis]
        act_sam_numdis = data_sam_numdis[:, 7:]
        act_sam_numdis = act_sam_numdis.flatten()
        
        M_list.append(np.nanmean(act_sam_numdis))
        STD_list.append(np.nanstd(act_sam_numdis))
        LEN_list.append(len(act_sam_numdis[~np.isnan(act_sam_numdis)]))
        SUM_list.append(np.nansum(act_sam_numdis))
        num_dis_list.append(num_dis)

num_dis_list = np.expand_dims(np.asarray(num_dis_list).flatten(), axis=1)
M_list = np.expand_dims(np.asarray(M_list).flatten(), axis=1)
STD_list= np.expand_dims(np.asarray(STD_list).flatten(), axis=1)
LEN_list = np.expand_dims(np.asarray(LEN_list).flatten(), axis=1)
SUM_list = np.expand_dims(np.asarray(SUM_list).flatten(), axis=1)

plot_data = np.concatenate((num_dis_list, M_list, STD_list, LEN_list, SUM_list), axis=1)
plot_data = pd.DataFrame(plot_data)
plot_data.columns = ['num_dis', 'M', 'STD', 'LEN', 'SUM']
plot_data_M = plot_data.groupby(plot_data['num_dis']).mean()['M']
plot_data_SUM = plot_data.groupby(plot_data['num_dis']).sum()['SUM']
plot_data_LEN = plot_data.groupby(plot_data['num_dis']).sum()['LEN']

plt.plot(plot_data_SUM / plot_data_LEN)
# plt.plot(plot_data_M)
