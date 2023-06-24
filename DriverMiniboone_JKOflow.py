
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
import numpy as np
import os
import h5py
import torch
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
sys.path.append('./src/train_subproblem.py')
sys.path.append('./src/utils.py')
from src.utils import *
from src.train_subproblem import *

gpu_device=1
file = './data/miniboone/data.npy'
xktrain, xkval, xktest = load_data_normalised(file)
n_dims = xktrain.shape[1]
print('xk.shape = ', xktrain.shape, ', xkval.shape = ', xkval.shape, ', xktest.shape =', xktest.shape)

xktrain = torch.FloatTensor(xktrain)
xkval = torch.FloatTensor(xkval)
xktest = torch.FloatTensor(xktest)

print('CONCATENATING VALIDATION AND TESTING SAMPLES')
xkval = torch.cat((xkval, xktest), dim=0)

file = './logger.txt'
with open(file, mode='a'): pass

cf = getconfig()

"""#---------------------------- Set Parameters ---------------------------#"""
if cf.gpu:
    n_samples    = int(xktrain.shape[0]) #number of data samples
    def_viz_freq = 200 #plotting frequency
    def_batch    = 2000 #batch size
    def_niter    = 8000 #number of iterations
    def_m        = 256 #network width
    n_subproblems= 10 #number of jko flow subproblems
    max_epochs   = int(500) #max epochs

else: # if no gpu on platform, assume debugging on a local cpu
    n_samples    = int(xktrain.shape[0])
    # n_samples    = int(6000)
    def_viz_freq = 200
    def_batch    = 2000
    def_niter    = 8000
    def_m        = 256
    n_subproblems= 10
    max_epochs   = int(500)

parser = argparse.ArgumentParser('OT-Flow')

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=16, help="number of time steps for validation")
"""#---------------------------- Choose fixed alpha (alphaC) ---------------------------#"""
parser.add_argument('--alph'  , type=str, default='1.0,50.0,1.0') #values for alpha: [alphaL, alphaC, alphaR]
parser.add_argument('--m'     , type=int, default=def_m) #network width
parser.add_argument('--nTh'   , type=int, default=2) #number timesteps

parser.add_argument('--lr'       , type=float, default=1e-3) #learning rate
parser.add_argument("--drop_freq", type=int  , default=0, help="how often to decrease learning rate; 0 lets the mdoel choose")
parser.add_argument("--lr_drop"  , type=float, default=10.0, help="how much to decrease learning rate (divide by)")
parser.add_argument('--weight_decay', type=float, default=0.0) #learning rate decay (placeholder)

parser.add_argument('--prec'      , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--niters'    , type=int, default=def_niter) #number of iterations
parser.add_argument('--batch_size', type=int, default=def_batch)
parser.add_argument('--test_batch_size', type=int, default=def_batch)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--early_stopping', type=int, default=20)

parser.add_argument('--save', type=str, default='experiments/cnf/large')
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=gpu_device)
parser.add_argument('-f')

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# decrease the learning rate based on validation
ndecs = 0
n_vals_wo_improve=0
args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

# get precision type
if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32

# get timestamp for saving models
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath='./logger.txt')
logger.info("start time: " + start_time)
logger.info(args)
# device = torch.device('cuda:1' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


log_msg = []
itr = 0
n_epochs = 0
n_subproblems_used = 0
weight_decay = 0

torch.set_default_dtype(prec)
cvt = lambda x: x.type(prec).to(device, non_blocking=True)

# neural network for the potential function Phi
d      = xktrain.shape[1]
lr     = args.lr
alph   = args.alph
nt     = args.nt
nt_val = args.nt_val
nTh    = args.nTh
m      = args.m
net = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
net = net.to(prec).to(device)

KL_hist = [] # history of KL divergence after each subproblem
MMD_hist = [] # history of MMD values after each subproblem

optim = torch.optim.Adam(net.parameters(), lr=args.lr)#, weight_decay=args.weight_decay ) # lr=0.04 good

logger.info(net)
logger.info("-------------------------")
logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
logger.info("-------------------------")
logger.info(str(optim)) # optimizer info
logger.info("data={:} batch_size={:} gpu={:}".format(xktrain, args.batch_size, args.gpu))
logger.info("maxIters={:} ".format(args.niters))
logger.info("saveLocation = {:}".format(args.save))
logger.info("-------------------------\n")

end = time.time()

bestParams = None
# setup data [nSamples, d]

print('n_samples = ', xktrain.shape[0])

costC_hist = []
costTotal_hist = []
costTotal_hists = [] 

current_dataset = TensorDataset(torch.FloatTensor(xktrain))
current_loader = DataLoader(dataset=current_dataset,
                          batch_size=args.batch_size, shuffle=False)
current_dataset_val = TensorDataset(xkval)
current_loader_val = DataLoader(dataset=current_dataset_val,
                          batch_size=xkval.shape[0], shuffle=False)

log_msg = (
    '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s}      {:9s}  {:9s}  {:9s}  {:9s}  '.format(
        'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'valLoss', 'valL', 'valC', 'valR'
    )
)
logger.info(log_msg)
time_meter = AverageMeter()
xkval_latest = xkval.clone()

#-------------------------------------------------------------------
# Compute MMD Loss before starting
#-------------------------------------------------------------------
start_time_MMD = time.time()
with torch.no_grad():
    sample_flow = cvt(torch.randn(6000,net.d))
    mmd_val = mmd(xkval, sample_flow[:,0:net.d].cpu())
    print('MMD of reconstruction = ', mmd_val)
    MMD_hist.append(mmd_val)
end_time_MMD = time.time()
print('Time to compute MMD = ', end_time_MMD - start_time_MMD)

for i in range(n_subproblems):
  net.to(device)
  net.train()
  loss_val, loss_vals, cost_L, cost_C, cost_R, opt_statedict, params, grad_norm, n_epochs_local = train_Phi(optim, 
                                                                                                            itr, 
                                                                                                            alph, 
                                                                                                            current_loader, 
                                                                                                            nt,
                                                                                                            def_batch,
                                                                                                            lr, 
                                                                                                            net, 
                                                                                                            max_epochs,
                                                                                                            weight_decay,
                                                                                                            outer_iter=i+1, 
                                                                                                            print_freq=10,
                                                                                                            plot_freq=5, 
                                                                                                            verbose=True, 
                                                                                                           val_loader = current_loader_val)
  lr = 1*lr
  print('learning rate', lr)
  time_meter.update(time.time() - end)
  n_epochs = n_epochs + n_epochs_local
  costC_hist.append(cost_C)
  costTotal_hist.append(loss_val)
  costTotal_hists.append(loss_vals)
  n_subproblems_used = n_subproblems_used + 1

  #--------------------------------------------------
  # save the weights after first subproblem
  #--------------------------------------------------
  state = {
    'model_state_dict': params,
  }

  net.to('cpu') # move net to CPU to generate new data

  save_dir = './experiments/' + 'miniboone_weights/'
  file_name = save_dir + 'miniboone_weights_iter_' + str(i+1) + '.pth'
  torch.save(state, file_name)
  print('\nModel weights saved to ' + file_name)       
  current_loader = create_data_loader(current_loader, net, args.nt)
  current_loader_val = create_data_loader(current_loader_val, net, args.nt)

  xkval_latest = next(iter(current_loader_val))[0]
  logger.info('Subproblem Training Time: {:} seconds'.format(time_meter.sum))
  logger.info('Subproblem_' + str(i + 1) +  '_Training Finished')

  net.to(device) # move net back to GPU (or desired device)

  #----------------------------------------------------------------------------------------------------------
  # Compute MMD of current subproblem
  # start from latest weights saved and flow them backwards to obtain MMD between samples of rho0hat and samples of rho0
  #----------------------------------------------------------------------------------------------------------
  with torch.no_grad():
    sample_flow = cvt(torch.randn(10000,net.d))
    count = n_subproblems_used
    for i in range(n_subproblems_used):
      file_name= save_dir + 'miniboone_weights_iter_' + str(count) + '.pth'
      state = torch.load(file_name, map_location=torch.device(device))
      net.load_state_dict(state['model_state_dict'])
      sample_flow = integrate(sample_flow[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()
      
      save_path = './experiments/miniboone/sample_flow_' + str(count) + '.png'
      count=count-1
      
      start_time_MMD = time.time()
      if i==n_subproblems_used-1:
        mmd_val = mmd(xkval, sample_flow[:,0:net.d].cpu())
        print('MMD of reconstruction = ', mmd_val)
        MMD_hist.append(mmd_val)
      end_time_MMD = time.time()
      print('Time to compute MMD = ', end_time_MMD - start_time_MMD)

logger.info("Training Time: {:} seconds".format(time_meter.sum))
logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(xkval,int(alph[1]),int(alph[2]),m))

