import os
import matplotlib.pyplot as plt
import numpy as np
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.core import Mask
from os.path import join as pjoin

# load pic to compute activation
main_path = 'nfs/s2/userhome/zhouming/workingdir/numerosity/stimulus' 
out_path = 'nfs/s2/userhome/zhouming/workingdir/numerosity/out/activation' 
stim_set = ['', '', '']
dnn = AlexNet()
layer = 'conv5'
chn = 'all'
mask = Mask(layer,[chn])
# start computing
for stim in stim_set:
    stim_path = pjoin(main_path, stim)
    pic_all = np.zeros(len(os.listdir(stim_path)), 3, 224, 224)
    for idx,pic_name in enumerate(os.listdir(stim_path)):
        pic_path = pjoin(stim_path, pic_name)
        pic = plt.imread(pic_path)
        pic_all[idx] = pic.transpose((2,0,1))
    act = dnn.compute_activation(pic_all, mask).get(layer)
    np.save(act, pjoin(out_path, f'{stim}_act.npy'))

# using activation to select the numerosity unit
    
    
