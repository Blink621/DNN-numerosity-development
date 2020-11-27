# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:57:48 2020

@author: 薛
"""


# %%
import scipy.io
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# %% 准备数据
X_data = np.load('act_fc1_relu.npy')
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

# %%
num_dset = 3
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

# %% 将数据划分为训练集、验证集和测试集
data = data_all
data_train_sam = data[data[:, 2] < 201]
data_train_com = data[(data[:, 2] > 200) & (data[:, 2] < 401)]
data_val_sam = data[(data[:, 2] > 400) & (data[:, 2] < 501)]
data_val_com = data[(data[:, 2] > 500)]
# data_test_sam = data[(data[:, 2] > 400) & (data[:, 2] < 501)]
# data_test_com = data[data[:, 2] > 500]

# %% 将图片两两配对，并且构成真正用到的模型数据
X_list = []
y_list = []
for sam in range(1, (num_dnum + 1)):
    for com in range(1, (num_dnum + 1)):
        X_sam = data_train_sam[data_train_sam[:, 1] == sam][:, 3:]
        X_com = data_train_com[data_train_com[:, 1] == com][:, 3:]
        X_sam_com = X_sam - X_com
        # X_com_sam = X_com - X_sam
        if (sam - com) > 0:
            X_list.append(X_sam_com)
            # X_list.append(X_com_sam)
            y = np.repeat(0, np.shape(X_sam)[0])
            y = np.expand_dims(y, axis=0)
            y_list.append(y.T)
        elif (sam - com) < 0:
            X_list.append(X_sam_com)
            # X_list.append(X_com_sam)
            y = np.repeat(1, np.shape(X_sam)[0])
            y = np.expand_dims(y, axis=0)
            y_list.append(y.T)

X_train = np.concatenate(tuple(X_list), axis=0)
y_train = np.concatenate(tuple(y_list), axis=0)

# %%
X_list = []
y_list = []
for sam in range(1, (num_dnum + 1)):
    for com in range(1, (num_dnum + 1)):
        X_sam = data_val_sam[data_val_sam[:, 1] == sam][:, 3:]
        X_com = data_val_com[data_val_com[:, 1] == com][:, 3:]
        X_sam_com = X_sam - X_com
        # X_com_sam = X_com - X_sam
        if (sam - com) > 0:
            X_list.append(X_sam_com)
            # X_list.append(X_com_sam)
            y = np.repeat(0, np.shape(X_sam)[0])
            y = np.expand_dims(y, axis=0)
            y_list.append(y.T)
        elif (sam - com) < 0:
            X_list.append(X_sam_com)
            # X_list.append(X_com_sam)
            y = np.repeat(1, np.shape(X_sam)[0])
            y = np.expand_dims(y, axis=0)
            y_list.append(y.T)

X_val = np.concatenate(tuple(X_list), axis=0)
y_val = np.concatenate(tuple(y_list), axis=0)

# %%
data_numneu_train = torch.tensor(X_train).float()
data_numneu_val = torch.tensor(X_val).float()

labels_train = torch.tensor(np.squeeze(y_train)).long()
labels_val = torch.tensor(np.squeeze(y_val)).long()

# %%
# scipy.io.savemat('fc1_train_and_val.mat', mdict={'X_train': X_train, 'X_val': X_val,})
# scipy.io.savemat('fc1_data_rest.mat', mdict={'X_data': X_data})

# %%
# n_numneu = len(neuid_rest)
n_numneu = len(neuid)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 1层全连接网络并且规定了对应的神经元的个数
        self.fc1 = nn.Linear(n_numneu, 2)
        self.fc2 = nn.Softmax(dim=1)
        # 构造Dropout方法，在每次训练过程中都随机“掐死”百分之二十的神经元，防止过拟合。
        # self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # 确保输入的tensor是展开的单列数据，把每张图片的通道、长度、宽度三个维度都压缩为一列
        # .view是干什么的？
        x = x.view(x.shape[0], -1)
        
        # x = self.dropout(F.relu(self.fc1(x)))
        
        # 在输出单元不需要使用Dropout方法
        x = self.fc1(x)
        x = self.fc2(x)  
        return x

# 对上面定义的Classifier类进行实例化
model = Classifier()

# 定义损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 优化方法为Adam梯度下降方法，学习率为0.003
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
train_losses, test_losses = [], []

print('开始训练')
# 对训练集的全部数据学习200遍，这个数字越大，训练时间越长
epochs = 100
for e in range(epochs):
    
    # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
    optimizer.zero_grad()
    
    # 对64张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
    
    # 这里还未读取data_numneu_train和labels_train
    log_ps = model(data_numneu_train)
    loss = criterion(log_ps, labels_train)
    loss.backward()
    optimizer.step()
    running_loss = loss.item()
    
    # 每次学完一遍数据集，都进行以下测试操作
    
    # 测试的时候不需要开自动求导和反向传播
    with torch.no_grad():
        # 关闭Dropout
        model.eval()
        
        # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
        
        # 这里还未读取data_numneu_val和labels_val
        ps = model(data_numneu_val)
        test_loss = criterion(ps, labels_val)
        
        # ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels_val.view(*top_class.shape)
        
        # 等号右边为每一批测试图片中预测正确的占比
        accuracytop1 = torch.mean(equals.type(torch.FloatTensor))
        
    # 恢复Dropout
    model.train()
    
    # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
    train_losses.append(running_loss)
    test_losses.append(test_loss)

    print("num_train: {}/{}.. ".format(e+1, epochs),
          "train_error: {:.3f}.. ".format(running_loss),
          "test_error: {:.3f}.. ".format(test_loss),
          "accuracytop1: {:.3f}".format(accuracytop1))

# %%
# 保存模型需要代码，torch.save
save_path = '/nfs/s2/userhome/xuezhichao/workingdir/group_coding/more_classifier_fc1_80.pth'
# save_path = 'E:/working/group_coding/more_classifier_fc1_80' + f'dset{dset}' + '.pth'
torch.save(model, save_path)

# %%
plt.plot(train_losses[3:], label='Training loss')
plt.plot(test_losses[3:], label='Validation loss')
plt.legend()
picname = '/nfs/s2/userhome/xuezhichao/workingdir/group_coding/more_classifier_fc1_80.png'
# picname = 'E:/working/group_coding/more_classifier_fc1_80' + f'dset{dset}' + '.png'
plt.savefig(picname)
plt.close()
