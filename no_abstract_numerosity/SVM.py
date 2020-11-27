# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:09:38 2020

@author: 薛
"""

import numpy as np
import scipy.io as io

# from load_data import load_stimuli, load_voxels
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pytorch_pretrained_biggan import (truncated_noise_sample)
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.core import Mask

# %% Prepare data

X_data = np.load('act_fc2_relu_test_simple.npy')
neuid = np.load('id90_fc2_relu.npy')
neuid = neuid[:, 0] - 1
neuid = neuid.astype(int)
X_data = X_data[:, neuid, :, :]
X_data = np.squeeze(X_data)
y_data = np.tile(np.repeat(range(1, 33), 600), 4)
choose_index = np.tile(np.tile(range(1, 601), 32), 4)
X_train = X_data[choose_index < 501]
y_train = y_data[choose_index < 501]
X_test = X_data[choose_index > 500]
y_test = y_data[choose_index > 500]


# %% Contruct decoding model

def biggan_generator(latent_space):
    """
    Parameters
    ----------
    latent_space : torch.Tensor
        shape(n_pic, n_feature)
    """
    #load model
    model = torch.load('/nfs/e2/workingshop/swap/models/biggan-deep-256.pkl')
    # define input
    truncation = 0.3
    pic_out = np.zeros((latent_space.shape[0], 3, 256, 256))
    # latent_array = latent_space.data.numpy()
    # print(latent_array.shape)
    for idx in range(latent_space.shape[0]):
        latent_single = latent_space[idx]
        class_vector = torch.Tensor(latent_single).unsqueeze(0)
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
        noise_vector = torch.Tensor(noise_vector)
        # compute
        output = model(noise_vector, class_vector, truncation)
        output = (output-output.min() )/ (output.max( )-output.min()) 
        pic_single = output[0].data.numpy()
        pic_out[idx] = pic_single
        # print(f'Finish loading pics:{idx}/{latent_space.shape[0]} in epoch{epoch}')
    return pic_out

n_components = 1  # number of neu
decode_net = nn.Sequential(
        nn.Linear(n_components, 32)
)

torch_com = lambda x: Variable(torch.from_numpy(x).double(), requires_grad=True)
optimizer = torch.optim.Adam(decode_net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

data_train = 1;
brain_train = torch.tensor(data_train).float()  # What is the data_train's form?
for epoch in range(20):  # How to realize epoch?
    latent_space = decode_net(brain_train)  # What is brain_train?
    latent_nor = nn.functional.normalize(latent_space, dim=1)
    pic_train = biggan_generator(latent_nor)
    #loss = loss_compute(loss_func, pic_train, pic_val)
    loss = loss_func(torch_com(pic_train), torch_com(pic_val_float))  # What is the pic_train and pic_val_float?
    print(f'loss:{loss.item()} in epoch{epoch}')
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# save model info
save_path = '/nfs/s2/userhome/zhouming/workingdir/Decode/decode_biggan.pth'
torch.save(decode_net, save_path)
