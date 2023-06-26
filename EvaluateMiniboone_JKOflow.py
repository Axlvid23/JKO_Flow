import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
import numpy as np
import os
# import h5py
from torch.nn.functional import pad
from matplotlib import colors # for evaluateLarge
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import math
from numbers import Number
import logging
import torch.nn as nn
import copy
import argparse
import time
import datetime
import torch.optim as optim
import sys
from src.utils import *

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

cvt = lambda x: x.type(prec).to(device, non_blocking=True)


# ------------------------------------------------------------------------------------------------------------------------
# Setup -- Make sure this is configured to be the same as what you ran in DriverMiniboone_JKOflow.py
# ------------------------------------------------------------------------------------------------------------------------

# neural network for the potential function Phi
d      = 43
alph   = [1, 1, 1]
nt_val = 16
nTh    = 2
m      = 256

net = Phi(nTh=nTh, m=m, d=d, alph=alph)
net = net.to(device)
data_name = 'miniboone'
iter = 'iter1'
file = 'alpha05_10subproblems'
n_subproblems = 10
alpha = '50'
file = './data/miniboone/data.npy'
prec = torch.float32

# load the data and print the shapes
xktrain, xkval, xktest = load_data_normalised(file)
n_dims = xktrain.shape[1]
print('xk.shape = ', xktrain.shape, ', xkval.shape = ', xkval.shape, ', xktest.shape =', xktest.shape)
xkval = torch.FloatTensor(xkval).to(device)
save_dir = './experiments/' + data_name + '/alpha' + alpha + '_' + str(n_subproblems) + 'subproblems/' + iter +'/'

# ---------------------------------------------------------------------------------------------------------------
# change the dims d1 and d2 to analyze 2d slice of Miniboone dataset
# in the paper we used 16 v 17 and 28 v 29.
# ---------------------------------------------------------------------------------------------------------------
d1 = 16
d2 = d1+1

with torch.no_grad():
    sample_flow = cvt(torch.randn(6000,net.d))
    rho1_samples = sample_flow.clone()
    mmd_hist = []
    mmd_val_2d = mmd(xkval[:, d1:d2].cpu(), sample_flow[:, d1:d2].cpu())
    mmd_hist.append(mmd_val_2d)
    forward_flow_samples = cvt(xkval)
    count = n_subproblems
    print('subproblems', n_subproblems)
    for i in range(n_subproblems):
      file_name = save_dir +  data_name + '_weights_iter_' + str(count) + '.pth'
      print('loading ', file_name)
      state = torch.load(file_name, map_location=torch.device(device))
      net.load_state_dict(state['model_state_dict'])
      sample_flow = integrate(sample_flow[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()
      mmd_val_2d = mmd(xkval[:, d1:d2].cpu(), sample_flow[:, d1:d2].cpu())
      print('2D MMD at subproblem, ', count, ',  = ', mmd_val_2d)
      mmd_hist.append(mmd_val_2d)
      count=count-1

    for i in range(n_subproblems):
      file_name = save_dir +  data_name + '_weights_iter_' + str(i+1) + '.pth'
      state = torch.load(file_name, map_location=torch.device(device))
      net.load_state_dict(state['model_state_dict'])
      forward_flow_samples = integrate(forward_flow_samples[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph).detach()

nBins = 66
LOWX  = -4
HIGHX = 4
LOWY  = -4
HIGHY = 4


fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 10)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# plot xk values from rho0
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
saveloc = './experiments/miniboone/fig.png'
im1 , _, _, map1 = axs.hist2d(xkval.detach().cpu().numpy()[:, d1], xkval.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
plt.axis('off')
plt.savefig(saveloc,bbox_inches='tight')

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 10)
im2 , _, _, map2 = axs.hist2d(sample_flow.detach().cpu().numpy()[:, d1], sample_flow.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
plt.axis('off')

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 10)
im2 , _, _, map2 = axs.hist2d(rho1_samples.detach().cpu().numpy()[:, d1], rho1_samples.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
plt.axis('off')

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 10)
im2 , _, _, map2 = axs.hist2d(forward_flow_samples.detach().cpu().numpy()[:, d1], forward_flow_samples.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
plt.axis('off')

from matplotlib.ticker import StrMethodFormatter
plt.plot(mmd_hist)
plt.xticks(np.arange(len(mmd_hist), step=10), np.arange(1, len(mmd_hist)+1, step=10))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}'))
plt.ylabel('2D MMD')
plt.xlabel('Subproblem')