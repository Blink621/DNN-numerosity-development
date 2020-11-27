# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:34:25 2020

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

# %% 定义载入模型的类
numlist = np.asarray(range(1, 33))
neuid = np.load('id80_fc1_relu.npy')
neuid = neuid[:, 0] - 1
neuid = neuid.astype(int)
n_numneu = len(neuid)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 1层全连接网络并且规定了对应的神经元的个数
        self.fc1 = nn.Linear(n_numneu, 32)
        # 构造Dropout方法，在每次训练过程中都随机“掐死”百分之二十的神经元，防止过拟合。
        # self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # 确保输入的tensor是展开的单列数据，把每张图片的通道、长度、宽度三个维度都压缩为一列
        # .view是干什么的？
        x = x.view(x.shape[0], -1)
        
        # x = self.dropout(F.relu(self.fc1(x)))
        
        # 在输出单元不需要使用Dropout方法
        x = F.log_softmax(self.fc1(x), dim=1)
        
        return x


# %% 准备数据
X_data = np.load('act_fc1_relu_test_simple.npy')
# neuid = np.load('id80_fc1_relu.npy')
# neuid = neuid[:, 0] - 1
# neuid = neuid.astype(int)
X_data = X_data[:, neuid, :, :]
X_data = np.squeeze(X_data)
y_data = np.tile(np.repeat(range(1, 33), 600), 4)
choose_index = np.tile(np.tile(range(1, 601), 32), 4)
# X_train = X_data[choose_index < 501]
# y_train = y_data[choose_index < 501]
# y_train = y_train - 1
X_test = X_data
y_test = y_data
y_test = y_test - 1
# data_numneu_train = torch.tensor(X_train).float()
data_numneu_test = torch.tensor(X_test).float()
# labels_train = torch.tensor(y_train).long()
labels_test = torch.tensor(y_test).long()

labels_test_top3 = np.repeat(np.expand_dims(y_test, axis=1), 3, axis=1)
labels_test_top3 = torch.tensor(labels_test_top3).long()

labels_test_top5 = np.repeat(np.expand_dims(y_test, axis=1), 5, axis=1)
labels_test_top5 = torch.tensor(labels_test_top5).long()


# %%
# scipy.io.savemat('fc1_test.mat', mdict={'X_test': X_test,})


# %% 测试模型

# 定义损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 载入模型
model = torch.load('num_classifier_fc1_80.pth')

# 开始测试，此时要关闭梯度的传播
with torch.no_grad():
    # 关闭Dropout
    model.eval()
    
    # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
    
    # 这里还未读取data_numneu_test和labels_test
    log_ps = model(data_numneu_test)
    test_loss = criterion(log_ps, labels_test)
    
    # 下面三行没看懂
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(3, dim=1)
    equals = top_class == labels_test_top3.view(*top_class.shape)
    
    # 等号右边为每一批测试图片中预测正确的占比
    accuracytop3 = torch.mean(equals.type(torch.FloatTensor)) * 3
    
    # 下面三行没看懂
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(5, dim=1)
    equals = top_class == labels_test_top5.view(*top_class.shape)
    
    # 等号右边为每一批测试图片中预测正确的占比
    accuracytop5 = torch.mean(equals.type(torch.FloatTensor)) * 5
    
    # 下面三行没看懂
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels_test.view(*top_class.shape)
    
    # 等号右边为每一批测试图片中预测正确的占比
    accuracytop1 = torch.mean(equals.type(torch.FloatTensor))
    
    print("accuracytop1: {:.3f}".format(accuracytop1),
          "accuracytop3: {:.3f}".format(accuracytop3),
          "accuracytop5: {:.3f}".format(accuracytop5))


# %%
possibility = ps.numpy()
num_label = labels_test.numpy() + 1
data = np.c_[num_label, possibility]
data = pd.DataFrame(data)
str_num = ['num']
for num in range(32):
    str_num.append('%d' % (num + 1))
data.columns = str_num


# %%
plt.figure(figsize=(20, 15))
df = data
cfm = []
for i in range(8):
    for j in range(4):
        PN = '%d' % (i * 4 + j + 1)
        dfm = df.groupby(df['num']).mean()[PN]

        dfstd = df.groupby(df['num']).std()[PN]
        dferr = dfstd / ((np.shape(data)[0] / 32) ** 0.5)
        ax = plt.subplot(8, 4, (i)*4+j+1)
        ax.set_xticks([])
        ax.set_title(f'PN = {PN}',fontsize=10)
        plt.ylim(0, 1)
        plt.errorbar(range(1, 33), dfm, yerr=dferr, fmt='k', ecolor='black', elinewidth=2)
        cfm.append(dfm.tolist())

cfm = np.asarray(cfm)
cfm = cfm.T
plt.matshow(cfm, cmap=plt.cm.gray)

row_sum = np.sum(cfm, axis=1)
err_matrix = cfm / row_sum
np.fill_diagonal(err_matrix, 0)
plt.matshow(err_matrix, cmap=plt.cm.gray)

picname = 'fc1_nn_test.png'
plt.savefig(picname)


# %%
y_pre_cfm = top_class.numpy()
y_test_cfm = labels_test.numpy()
cfm = confusion_matrix(y_test_cfm, y_pre_cfm)
plt.matshow(cfm, cmap=plt.cm.gray)

row_sum = np.sum(cfm, axis=1)
err_matrix = cfm / row_sum
np.fill_diagonal(err_matrix, 0)
plt.matshow(err_matrix, cmap=plt.cm.gray)

# %%
for dset in range(1):
    
    X_test_dset = X_test[dset*19200 : (dset+1)*19200, :]
    
    data_numneu_test = torch.tensor(X_test_dset).float()
    
    y_test_dset = y_test[dset*19200 : (dset+1)*19200]
    
    labels_test = torch.tensor(y_test_dset).long()

    labels_test_top3 = np.repeat(np.expand_dims(y_test_dset, axis=1), 3, axis=1)
    labels_test_top3 = torch.tensor(labels_test_top3).long()
    
    labels_test_top5 = np.repeat(np.expand_dims(y_test_dset, axis=1), 5, axis=1)
    labels_test_top5 = torch.tensor(labels_test_top5).long()
    # 开始测试，此时要关闭梯度的传播
    with torch.no_grad():
        # 关闭Dropout
        model.eval()
        
        # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
        
        # 这里还未读取data_numneu_test和labels_test
        log_ps = model(data_numneu_test)
        test_loss = criterion(log_ps, labels_test)
        
        # 下面三行没看懂
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(3, dim=1)
        equals = top_class == labels_test_top3.view(*top_class.shape)
        
        # 等号右边为每一批测试图片中预测正确的占比
        accuracytop3 = torch.mean(equals.type(torch.FloatTensor)) * 3
        
        # 下面三行没看懂
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(5, dim=1)
        equals = top_class == labels_test_top5.view(*top_class.shape)
        
        # 等号右边为每一批测试图片中预测正确的占比
        accuracytop5 = torch.mean(equals.type(torch.FloatTensor)) * 5
        
        # 下面三行没看懂
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels_test.view(*top_class.shape)
        
        # 等号右边为每一批测试图片中预测正确的占比
        accuracytop1 = torch.mean(equals.type(torch.FloatTensor))
        
        print("accuracytop1: {:.3f}".format(accuracytop1),
              "accuracytop3: {:.3f}".format(accuracytop3),
              "accuracytop5: {:.3f}".format(accuracytop5))
    
    
    # %%
    possibility = ps.numpy()
    num_label = labels_test.numpy() + 1
    data = np.c_[num_label, possibility]
    data = pd.DataFrame(data)
    str_num = ['num']
    for num in range(32):
        str_num.append('%d' % (num + 1))
    data.columns = str_num
    
    
    # %%
    plt.figure(figsize=(20, 15))
    df = data
    cfm = []
    for i in range(8):
        for j in range(4):
            PN = '%d' % (i * 4 + j + 1)
            dfm = df.groupby(df['num']).mean()[PN]
    
            dfstd = df.groupby(df['num']).std()[PN]
            dferr = dfstd / ((np.shape(data)[0] / 32) ** 0.5)
            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}',fontsize=10)
            plt.ylim(0, 1)
            plt.errorbar(range(1, 33), dfm, yerr=dferr, fmt='k', ecolor='black', elinewidth=2)
            cfm.append(dfm.tolist())
    
    cfm = np.asarray(cfm)
    cfm = cfm.T
    plt.matshow(cfm, cmap=plt.cm.gray)
    
    row_sum = np.sum(cfm, axis=1)
    err_matrix = cfm / row_sum
    np.fill_diagonal(err_matrix, 0)
    plt.matshow(err_matrix, cmap=plt.cm.gray)

    picname = 'fc1_nn_test.png'
    # plt.savefig(picname)
    
    
    # %%
    y_pre_cfm = top_class.numpy()
    y_test_cfm = labels_test.numpy()
    cfm = confusion_matrix(y_test_cfm, y_pre_cfm)
    plt.matshow(cfm, cmap=plt.cm.gray)
    picname = 'fc1_nn_test' + str(dset) + '_cfmhd.png'
    plt.savefig(picname)
    
    row_sum = np.sum(cfm, axis=1)
    err_matrix = cfm / row_sum
    np.fill_diagonal(err_matrix, 0)
    plt.matshow(err_matrix, cmap=plt.cm.gray)
    picname = 'fc1_nn_test' + str(dset) + '_cfmnd.png'
    # plt.savefig(picname)
    
    # %%
    df = data
    
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
            PN = '%d' % (i * 4 + j + 1)
            dfm = df.groupby(df['num']).mean()[PN]
            dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
    
            ax = plt.subplot(8, 4, (i)*4+j+1)
            ax.set_xticks([])
            ax.set_title(f'PN = {PN}', fontsize=10)
            plt.ylim(-0.1, 1.2)

            ax.plot(dfmn)
            if len(dfm) != 0:
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
