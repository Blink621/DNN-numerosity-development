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
from os.path import join as pjoin

act_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/activation' 
num_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/numerosity_unit' 
model_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/model/linear' 

# %% 准备数据
X_data = np.load(pjoin(act_path, 'act_fc2_relu.npy'))
neuid = np.load(pjoin(num_path, 'id80_fc2_relu.npy'))
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

# X_data = X_data[:, neuid, :, :]
X_data = X_data[:, neuid_rest, :, :]
X_data = np.squeeze(X_data)
y_data = np.tile(np.repeat(range(1, 33), 600), 3)
choose_index = np.tile(np.tile(range(1, 601), 32), 3)
X_train = X_data[choose_index < 501]
y_train = y_data[choose_index < 501]
y_train = y_train - 1
X_val = X_data[choose_index > 500]
y_val = y_data[choose_index > 500]
y_val = y_val - 1
data_numneu_train = torch.tensor(X_train).float()
data_numneu_val = torch.tensor(X_val).float()
labels_train = torch.tensor(y_train).long()
labels_val = torch.tensor(y_val).long()

labels_val_top3 = np.repeat(np.expand_dims(y_val, axis=1), 3, axis=1)
labels_val_top3 = torch.tensor(labels_val_top3).long()

labels_val_top5 = np.repeat(np.expand_dims(y_val, axis=1), 5, axis=1)
labels_val_top5 = torch.tensor(labels_val_top5).long()


# %%
# scipy.io.savemat('fc1_train_and_val.mat', mdict={'X_train': X_train, 'X_val': X_val,})
# scipy.io.savemat('fc2_data_rest.mat', mdict={'X_data': X_data})

# %%
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
epochs = 200
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
        log_ps = model(data_numneu_val)
        test_loss = criterion(log_ps, labels_val)
        
        # 下面三行没看懂
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(3, dim=1)
        equals = top_class == labels_val_top3.view(*top_class.shape)
        
        # 等号右边为每一批64张测试图片中预测正确的占比
        accuracytop3 = torch.mean(equals.type(torch.FloatTensor)) * 3
        
        # 下面三行没看懂
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(5, dim=1)
        equals = top_class == labels_val_top5.view(*top_class.shape)
        
        # 等号右边为每一批64张测试图片中预测正确的占比
        accuracytop5 = torch.mean(equals.type(torch.FloatTensor)) * 5
        
        # 下面三行没看懂
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels_val.view(*top_class.shape)
        
        # 等号右边为每一批64张测试图片中预测正确的占比
        accuracytop1 = torch.mean(equals.type(torch.FloatTensor))
        
    # 恢复Dropout
    model.train()
    
    # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
    train_losses.append(running_loss)
    test_losses.append(test_loss)

    print("num_train: {}/{}.. ".format(e+1, epochs),
          "train_error: {:.3f}.. ".format(running_loss),
          "test_error: {:.3f}.. ".format(test_loss),
          "accuracytop1: {:.3f}".format(accuracytop1),
          "accuracytop3: {:.3f}".format(accuracytop3),
          "accuracytop5: {:.3f}".format(accuracytop5))


# %%
# 保存模型需要代码，torch.save
# save_path = '/nfs/s2/userhome/xuezhichao/workingdir/group_coding/num_classifier.pth'
save_path = pjoin(model_path, 'num_classifier_fc2_80.pth')
torch.save(model, save_path)


# %%
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend()
picname = pjoin(model_path, 'num_classifier_fc2_80.jpg')
plt.savefig(picname)
