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


# %% 准备数据
X_data = np.load('act_fc2_relu_test_simple.npy')
neuid = np.load('id80_fc2_relu.npy')
neuid = neuid[:, 0] - 1
neuid = neuid.astype(int)

neuid_all = np.arange(0, 4096)
neuid_rest_index = np.zeros(4096)

for num1 in neuid_all:
    for num2 in neuid:
        if num1 == num2:
            neuid_rest_index[num1] = 1
neuid_rest_index = 1 - neuid_rest_index
neuid_rest_index = neuid_rest_index.astype('bool')
neuid_rest = neuid_all[neuid_rest_index]

X_data = X_data[:, neuid_rest, :, :]
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


# %% 定义载入模型的类
n_numneu = len(neuid_rest)
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



# %%
scipy.io.savemat('fc2_rest_test.mat', mdict={'X_test': X_test,})


# %% 测试模型

# 定义损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 载入模型
model = torch.load('num_classifier_fc2_80_rest.pth')

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

picname = 'fc2_rest_nn_test.png'
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
for dset in range(4):
    
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

    picname = 'fc2_rest_nn_test.png'
    plt.savefig(picname)
    
    
    # %%
    y_pre_cfm = top_class.numpy()
    y_test_cfm = labels_test.numpy()
    cfm = confusion_matrix(y_test_cfm, y_pre_cfm)
    plt.matshow(cfm, cmap=plt.cm.gray)
    picname = 'fc2_rest_nn_test' + str(dset) + '_cfmhd.png'
    plt.savefig(picname)
    
    row_sum = np.sum(cfm, axis=1)
    err_matrix = cfm / row_sum
    np.fill_diagonal(err_matrix, 0)
    plt.matshow(err_matrix, cmap=plt.cm.gray)
    picname = 'fc2_rest_nn_test' + str(dset) + '_cfmnd.png'
    plt.savefig(picname)