# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:45:33 2020

@author: 薛
"""

import scipy.io
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats

def gaussian(x, *param):
    return param[0] * np.exp(-(x - param[1])**2 / (2 * param[2]**2))

# %% 准备数据
X_data = np.load('act_fc1_relu_test_simple.npy')
neuid = np.load('id80_fc1_relu.npy')
neuid = neuid[:, 0] - 1
neuid = neuid.astype(int)
neuid_all = np.arange(0, 4096)
neuid_rest_index = np.zeros(4096)

# %%
for num1 in neuid_all:
    for num2 in neuid:
        if num1 == num2:
            neuid_rest_index[num1] = 1
neuid_rest_index = 1 - neuid_rest_index
neuid_rest_index = neuid_rest_index.astype('bool')
neuid_rest = neuid_all[neuid_rest_index]

# %% 类的定义
# n_numneu = len(neuid_rest)
n_numneu = len(neuid)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(n_numneu, 2)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)
        self.fc4 = nn.Softmax(dim=1)
        # 构造Dropout方法，在每次训练过程中都随机“掐死”神经元，防止过拟合。
        self.dropout = nn.Dropout(p=0.75)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        x = self.fc1(self.dropout(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
# %%
num_dset = 4
num_dnum = 32
num_dindex = 600
X_data = X_data[:, neuid, :, :]
# X_data = X_data[:, neuid_rest, :, :]
X_data = np.squeeze(X_data)
dset = np.repeat(range(num_dset), num_dindex * num_dnum)
dset = np.expand_dims(dset, axis=1)
dnum = np.tile(np.repeat(range(1, (num_dnum + 1)), num_dindex), num_dset)
dnum = np.expand_dims(dnum, axis=1)
dindex = np.tile(np.tile(range(1, (num_dindex + 1)), num_dnum), num_dset)
dindex = np.expand_dims(dindex, axis=1)
data_all = np.concatenate((dset, dnum, dindex, X_data), axis=1)

# %%
# 定义损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 载入模型
model = torch.load('match_classifier_fc1_80.pth')

# %% 
for dset in range(num_dset):
    data = data_all[data_all[:, 0] == dset, :]
    data_test_sam = data[(data[:, 2] > 400) & (data[:, 2] < 501)]
    data_test_com = data[(data[:, 2] > 500)]
    
    # %% 将图片两两配对，并且构成真正用到的模型数据
    X_sam_row_list = []
    num_dis_list_all = []
    X_list = []
    y_list = []
    for sam in range(1, (num_dnum + 1)):
        for com in range(1, (num_dnum + 1)):
            X_sam_row = data_test_sam[data_test_sam[:, 1] == sam]
            X_com_row = data_test_com[data_test_com[:, 1] == com]
            X_sam = X_sam_row[:, 3:]
            X_com = X_com_row[:, 3:]
            X_sam_row_list.append(X_sam_row)
            X_sam_com = X_sam - X_com
            
            num_dis = sam - com
            num_dis = np.expand_dims(np.repeat(num_dis, len(X_sam_com)), axis=1)
            X_list.append(X_sam_com)
            num_dis_list_all.append(num_dis)
            
            if sam == com:
                y = np.repeat(0, np.shape(X_sam)[0])
            else:
                y = np.repeat(1, np.shape(X_sam)[0])
            y = np.expand_dims(y, axis=0)
            y_list.append(y.T)
    
    X_test = np.concatenate(tuple(X_list), axis=0)
    y_test = np.concatenate(tuple(y_list), axis=0)
    num_dis_list_all = np.concatenate(tuple(num_dis_list_all), axis=0)
    
    # %%
    data_numneu_test = torch.tensor(X_test).float()
    
    labels_test = torch.tensor(np.squeeze(y_test)).long()

    # %% 测试模型
    
    # 开始测试，此时要关闭梯度的传播
    with torch.no_grad():
        # 关闭Dropout
        model.eval()
        
        # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
        
        # 这里还未读取data_numneu_test和labels_test
        ps = model(data_numneu_test)
        test_loss = criterion(ps, labels_test)
        
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels_test.view(*top_class.shape)
        
        # 等号右边为每一批测试图片中预测正确的占比
        accuracytop1 = torch.mean(equals.type(torch.FloatTensor))
        
        print("accuracytop1: {:.3f}".format(accuracytop1))
        
    result = np.squeeze(top_class.numpy())
    labels = np.squeeze(y_test)
    acc_row = (result == labels)
    confusion_matrix = np.zeros((num_dnum, num_dnum))
    step = round(len(acc_row) / (num_dnum * num_dnum))
    num_dis_list = []
    num_ratio_list = []
    acc_list = []
    
    flag = 0
    for sam in range(num_dnum):
        for com in range(num_dnum):           
            begin_index = flag * step
            end_index = (flag + 1) * step
            acc = np.sum(acc_row[begin_index:end_index]) / step
            confusion_matrix[sam, com] = acc
            num_dis = sam - com
            num_dis_list.append(num_dis)
            acc_list.append(acc)
            num_ratio_list.append(((sam + 1) / (com + 1)))
            flag += 1
            
    plt.figure(figsize=(20, 15))
    plt.matshow(confusion_matrix, cmap=plt.cm.gray)
    
    df = pd.DataFrame(np.asarray([num_dis_list, acc_list, num_ratio_list]).T)
    df.columns = ['numdis', 'acc', 'numratio']
    
    dfg = df.groupby(df['numdis'])['acc']
    x = np.unique(df['numdis'])
    y = dfg.mean()
    err = dfg.std() / np.sqrt(dfg.count())
    plt.figure(figsize=(20, 15))
    plt.errorbar(x, y, yerr=err, fmt='g', ecolor='green', elinewidth=2)
    
    x = np.log2(df['numratio'])
    y = df['acc']
    plt.figure(figsize=(20, 15))
    plt.scatter(x, y)
    
    # %%
    numlist = np.asarray(range(1, 33))
    confusion_matrix = 1 - confusion_matrix
    plt.figure(figsize=(20, 15))
    for i in range(8):
        for j in range(4):
            PN = '%d' % (i * 4 + j)
            dfm = confusion_matrix[:, int(PN)]

            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}',fontsize=10)
            plt.ylim(0, 1)
            plt.plot(numlist, dfm)
    
    # %%
    sigmaline = []
    sigmalog = []
    sigmapow2 = []
    sigmapow3 = []
    r2line = []
    r2log = []
    r2pow2 = []
    r2pow3 = []
    plt.figure(figsize=(20, 15))
    for i in range(8):
        for j in range(4):
            PN = '%d' % (i * 4 + j)
            dfmn = confusion_matrix[:, int(PN)]
    
            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}', fontsize=10)
            plt.ylim(-0.1, 1.2)
            
            numlist = np.asarray(range(1, 33))
            numlist = np.delete(numlist, int(PN))
            dfmn = np.delete(dfmn, int(PN))
            ax.plot(numlist, dfmn)
            
            popt, pcov = curve_fit(gaussian, numlist, dfmn, p0=[1, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
            plt.plot(numlist, gaussian(numlist, *popt))
            sigmaline.append(popt[2])
            r2 = r2_score(dfmn, gaussian(numlist, *popt))
            r2line.append(r2)
    
            popt, pcov = curve_fit(gaussian, np.log2(numlist), dfmn, p0=[1, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
            sigmalog.append(popt[2])
            r2 = r2_score(dfmn, gaussian(np.log2(numlist), *popt))
            r2log.append(r2)
    
            popt, pcov = curve_fit(gaussian, numlist**(1/2), dfmn, p0=[1, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
            sigmapow2.append(popt[2])
            r2 = r2_score(dfmn, gaussian(numlist**(1/2), *popt))
            r2pow2.append(r2)
    
            popt, pcov = curve_fit(gaussian, numlist**(1/3), dfmn, p0=[1, 15, 15], bounds=([0, 0, 0], [np.inf, 33, np.inf]), maxfev=5000000)
            sigmapow3.append(popt[2])
            r2 = r2_score(dfmn, gaussian(numlist**(1/3), *popt))
            r2pow3.append(r2)
    picname = f'{dset+1}_gaussian_' + '.png'
    # plt.savefig(picname)
    # plt.close('all')

    # %%

    plt.figure(figsize=(20, 15))
    ax = plt.subplot(111)
    plt.tick_params(labelsize=20)
    numlist = np.asarray(range(1, 33))
    num = numlist
    ax.set_xticks(num)
    plt.plot(num, np.asarray(np.abs(sigmaline)) / (2**0.5), label = 'line')
    plt.plot(num, np.asarray(np.abs(sigmalog)) / (2**0.5), label = 'log')
    plt.plot(num, np.asarray(np.abs(sigmapow2)) / (2**0.5), label = 'pow1/2')
    plt.plot(num, np.asarray(np.abs(sigmapow3)) / (2**0.5), label = 'pow1/3')
    plt.xlabel('PN', fontsize=30)
    plt.ylabel('sigma', fontsize=30)
    ax.legend(fontsize=30)
    # picname = f'{num_of_set}_sigma_' + name + '.png'
    # plt.savefig(picname)
    # plt.close('all')

    # %%
    length = len(num)
    plt.figure(figsize=(20, 15))
    mat = np.zeros((length * 4, 2))
    mat[:,0] = r2line + r2log + r2pow2 + r2pow3
    mat[:,1] = ['1'] * length + ['2'] * length + ['3'] * length + ['4'] * length
    dfr2 = pd.DataFrame(mat)
    dfr2.columns = ['r2', 'scale']
    formula = 'r2~C(scale)'
    res = anova_lm(ols(formula, dfr2).fit())
    print(res);

    r2 = dfr2.groupby(dfr2['scale']).mean()['r2']
    r2se = dfr2.groupby(dfr2['scale']).std()['r2'] / length**0.5

    plt.ylim(0.6, 1.2)
    plt.tick_params(labelsize=30)
    plt.xlabel('Scale', fontsize=40)
    plt.ylabel('Goodness of fit (r2)', fontsize=40)
    plt.bar(['line', 'log', 'pow1/2', 'pow1/3'], r2, yerr=r2se, error_kw={'capsize' : 6})

    res1 = stats.ttest_rel(dfr2[dfr2['scale']==1]['r2'], dfr2[dfr2['scale']==2]['r2'])
    res2 = stats.ttest_rel(dfr2[dfr2['scale']==1]['r2'], dfr2[dfr2['scale']==3]['r2'])
    res3 = stats.ttest_rel(dfr2[dfr2['scale']==1]['r2'], dfr2[dfr2['scale']==4]['r2'])
    print('log', res1[1], 'pow1/2', res2[1], 'pow1/3', res3[1])
    # picname = f'{num_of_set}_r2_' + name + '.png'
    # plt.savefig(picname)
    # plt.close('all')
    
    # %%
    X_sam_list = np.concatenate(X_sam_row_list)
    X_sam_list_act = X_sam_list[:, 3:]
    
    axis_num = 0
    X_sam_list_max = np.expand_dims(np.max(X_sam_list_act, axis=axis_num), axis=1).T
    X_sam_list_min = np.expand_dims(np.min(X_sam_list_act, axis=axis_num), axis=1).T
    
    # axis_num = 1
    # X_sam_list_max = np.expand_dims(np.max(X_sam_list_act, axis=axis_num), axis=1)
    # X_sam_list_min = np.expand_dims(np.min(X_sam_list_act, axis=axis_num), axis=1)
    
    X_sam_list_act_nom = (X_sam_list_act - X_sam_list_min) / (X_sam_list_max - X_sam_list_min)
    # X_sam_list_act_nom[np.isnan(X_sam_list_act_nom)] = 0
    acc_row = np.squeeze(acc_row)
    acc_row = np.expand_dims(acc_row, axis=1)
    X_sam_list = np.concatenate((acc_row, X_sam_list[:, 1:2], num_dis_list_all, X_sam_list_act_nom), axis=1)
    
    plt.figure(figsize=(20, 15))
    for i in range(2):
        data = X_sam_list[X_sam_list[:, 0] == i, 1:]
        mean_list = []
        err_list = []
        for sam in range(1, (num_dnum + 1)):
            neuid_sam = neuid[neuid == sam] + 1
            data_sam = data[data[:, 0] == sam, neuid_sam]
            for com in range(1, (num_dnum + 1)):
                
                num_dis = sam - com
                
                data_one = data[data[:, 0] == num_dis, 1:].flatten()
                mean_list.append(np.nanmean(data_one))
                value_df = np.count_nonzero(~np.isnan(data_one))
                err_list.append(np.nanstd(data_one) / np.sqrt(value_df))
            x = num_dis; y = mean_list; err = err_list;
            plt.errorbar(x, y, yerr=err, elinewidth=4)
    input()
