# TrainToy_JKOflow.py
# training driver for the two-dimensional toy problems
import json
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
import argparse
import datetime
import sys
sys.path.append('./src/ImplicitOT')
from src.train_subproblem import *

prec = torch.float32
device = 'cuda'
cvt = lambda x: x.type(prec).to(device, non_blocking=True)
file = './logger.txt'
with open(file, mode='a'): pass
cf = getconfig()

"""#---------------------------- Set Parameters ---------------------------#"""
if cf.gpu: # if gpu on platform
    n_samples    = int(6000) #number of data samples
    def_batch    = 2000 #batch size
    def_niter    = 1500 #number of iterations
    n_subproblems  = 5 #number of JKO flow iterations
    max_epochs   = int(1e3) #maximum number of epochs
    def_viz_freq = max_epochs #setting plot frequency to once per subproblem

else:  # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 100
    def_batch    = 2000
    def_niter    = 1000
    n_subproblems  = 1
    max_epochs   = int(2.5e2)

"""#---------------------------- Choose 2d Toy problem ---------------------------#"""
### Select toy problem by choosing default dataset name
parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='swissroll'
)

"""#---------------------------- More important hyperparams ---------------------------#"""

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=8, help="number of time steps for validation")
"""#---------------------------- Choose fixed alpha (alphaC) ---------------------------#"""
parser.add_argument('--alph'  , type=str, default='1.0,10.0,0.5')  #values for alpha: [alphaL, alphaC, alphaR]
parser.add_argument('--m'     , type=int, default=16) #network width
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--n_samples'    , type=int  , default=n_samples)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lr'          , type=float, default=1e-2) #learning rate
parser.add_argument("--drop_freq"   , type=int  , default=100, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=1.0)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam']) #default optimizer
parser.add_argument('--prec'        , type=str  , default='single', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments/cnf/toy') #where to put the logger
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu'     , type=int, default=0)
parser.add_argument('--sample_freq', type=int, default=5)
parser.add_argument('-f')

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

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

"""#---------------------------- JKO flow Model Training ---------------------------#"""
log_msg = []
itr = 0
n_epochs = 0
n_subproblems_used = 0

torch.set_default_dtype(prec)
cvt = lambda x: x.type(prec).to(device, non_blocking=True)

# neural network for the potential function Phi
d      = 2
lr     = args.lr
alph   = args.alph
nt     = args.nt
nt_val = args.nt_val
nTh    = args.nTh
m      = args.m
net = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
net = net.to(prec).to(device)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {pytorch_total_params}')

optim = torch.optim.Adam(net.parameters(), lr=args.lr)#, weight_decay=args.weight_decay ) # lr=0.04 good

logger.info(net)
logger.info("-------------------------")
logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
logger.info("-------------------------")
logger.info(str(optim)) # optimizer info
logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
logger.info("saveLocation = {:}".format(args.save))
logger.info("-------------------------\n")

end = time.time()
bestParams = None
# setup data [nSamples, d]
# use one batch as the entire data set
xk = inf_train_gen(args.data, batch_size=args.n_samples)
xk = cvt(torch.from_numpy(xk))

print('n_samples = ', xk.shape[0])

#--------------------------------
# Plotting
#--------------------------------
plt.plot(xk[0:50,0].cpu(), xk[0:50,1].cpu(), 'ro')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.show()
print(xk[0:3,:])
print('xk.shape = ', xk.shape)

costC_hist = []
costTotal_hist = []
costTotal_hists = []

"""#---------------------------- Generate Data For Each Iteration ---------------------------#"""

current_dataset = TensorDataset(xk)

current_loader = DataLoader(dataset=current_dataset,
                          batch_size=args.batch_size, shuffle=False)

xkval = inf_train_gen(args.data, batch_size=args.n_samples)
xkval = cvt(torch.from_numpy(xkval))

current_dataset_val = TensorDataset(xkval)

current_loader_val = DataLoader(dataset=current_dataset_val,
                          batch_size=xkval.shape[0], shuffle=False)

log_msg = (
    '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s}      {:9s}  {:9s}  {:9s}  {:9s}  '.format(
        'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'valLoss', 'valL', 'valC', 'valR'
    )
)
logger.info(log_msg)
"""#---------------------------- The subproblem loop for fixed alphaC ---------------------------#"""

time_meter = AverageMeter()
for i in range(n_subproblems):
  net.to(device)
  loss_val, loss_vals, cost_L, cost_C, cost_R, xk, opt_statedict, params, grad_norm, n_epochs_local = train_Phi(optim, itr, alph, current_loader, nt, def_batch, lr, net, max_epochs, outer_iter=i+1,
                                                                        weight_decay=0, print_freq=1, plot_freq=args.viz_freq, verbose=True, val_loader = current_loader_val)
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
    'model_state_dict': net.state_dict(),
  }

  file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_' + str(i+1) + '.pth'
  makedirs('./experiments/' + args.data + '_weights/')
  torch.save(state, file_name)
  print('\nModel weights saved to ' + file_name)
  current_loader = create_data_loader(current_loader, net, args.nt)
  current_loader_val = create_data_loader(current_loader_val, net, args.nt)


  xkval = next(iter(current_loader_val))[0]
  logger.info('Subproblem Training Time: {:} seconds'.format(time_meter.sum))
  logger.info('Subproblem_' + str(i + 1) +  '_Training Finished')

  #--------------------------------------------------
  # Compute MMD of current subproblem
  #--------------------------------------------------
  sample_flow = cvt(torch.randn(6000,net.d))
  true_samples = inf_train_gen(args.data, batch_size=args.n_samples)
  true_samples = cvt(torch.from_numpy(true_samples))
  count = n_subproblems_used

  for i in range(n_subproblems_used):
    file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_' + str(count) + '.pth'
    state = torch.load(file_name, map_location=torch.device(device))
    net.load_state_dict(state['model_state_dict'])
    sample_flow = integrate(sample_flow[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()

    save_path = './experiments/' + args.data + '/sample_flow_' + str(count) + '.png'
    count=count-1
    if i==n_subproblems_used-1:
      mmd_val = mmd(true_samples.cpu(), sample_flow[:,0:2].cpu())
      print('MMD of reconstruction = ', mmd_val)

logger.info("Training Time: {:} seconds".format(time_meter.sum))
logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))

len(loss_vals)
plt.plot(np.arange(0,max_epochs), np.log(loss_vals))
plt.ylabel('log loss vals')
plt.xlabel('iterations x time steps')
print('loss function minimum iteration', np.argmin(loss_vals[10:]))
print('loss function minimum value', np.min(loss_vals[10:]))
plt.title('minimum iteration: {}'.format(np.argmin(loss_vals[10:])))

#--------------------------------------------------------
# function for plotting samples
#--------------------------------------------------------
def save_samples_plot(samples, save_path):
  fig = plt.figure(figsize=(7, 7))
  ax = plt.subplot(1, 1, 1, aspect="equal")
  vmin = 0.0; vmax = 1.0

  nBins = 80
  LOWX  = -4
  HIGHX = 4
  LOWY  = -4
  HIGHY = 4

  extent = [[LOWX, HIGHX], [LOWY, HIGHY]]
  h3, _, _, map3 = ax.hist2d(samples.detach().cpu().numpy()[:, 0], samples.detach().cpu().numpy()[:, 1],
            range=extent, bins=nBins)
  h3 = h3/(samples.shape[0])
  im3 = ax.imshow(h3)
  im3.set_clim(vmin, vmax)
  ax.axis('off')
  plt.savefig(save_path,bbox_inches='tight')
  plt.show()
  plt.close(fig)
  print("finished plotting to folder", save_path)

mmds = []

#--------------------------------------------------------
# flow all at once:
#--------------------------------------------------------
sample_flow = cvt(torch.randn(6000,net.d))
true_samples = inf_train_gen(args.data, batch_size=args.n_samples)
true_samples = cvt(torch.from_numpy(true_samples))
count = n_subproblems_used
for i in range(n_subproblems_used):
  file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_' + str(count) + '.pth'
  state = torch.load(file_name,
                       map_location=torch.device(device))
  net.load_state_dict(state['model_state_dict'])
  sample_flow = integrate(sample_flow[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()

  save_path = './experiments/sample_flow_' + str(count) + '.png'
  save_samples_plot(sample_flow, save_path)
  count=count-1
  mmds.append(mmd(true_samples.cpu(), sample_flow[:,0:2].cpu()))
  if i==n_subproblems_used-1:
    mmd_val = mmd(true_samples.cpu(), sample_flow[:,0:2].cpu())
    print('MMD of reconstruction = ', mmd_val)

data_name = args.data
with open('./experiments/alpha_' + str(args.alph[1]) + ".txt", 'w') as f:
    json.dump(mmds, f, indent=2)

"""### Specific for 5 subproblems"""
y = cvt(torch.randn(1000,net.d))
file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_5.pth'
state = torch.load(file_name,
                       map_location=torch.device(device))
net.load_state_dict(state['model_state_dict'])
# apply backward integration using latest subproblem weights
sample_flow1 = integrate(y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()

file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_4.pth'

state = torch.load(file_name,
                       map_location=torch.device(device))
net.load_state_dict(state['model_state_dict'])
sample_flow2 = integrate(sample_flow1[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()

file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_3.pth'

state = torch.load(file_name,
                       map_location=torch.device(device))
net.load_state_dict(state['model_state_dict'])
sample_flow3 = integrate(sample_flow2[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()

file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_2.pth'
state = torch.load(file_name,
                       map_location=torch.device(device))
net.load_state_dict(state['model_state_dict'])
sample_flow4 = integrate(sample_flow3[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()

file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_1.pth'
state = torch.load(file_name,
                       map_location=torch.device(device))
net.load_state_dict(state['model_state_dict'])
sample_flow5 = integrate(sample_flow4[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph).detach()

nBins = 70
LOWX  = -4
HIGHX = 4
LOWY  = -4
HIGHY = 4

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
fig.suptitle('Inverse flow for {:d} subproblems'.format(n_subproblems))
im1 , _, _, map1 = axs[0, 0].hist2d(y.detach().cpu().numpy()[:, 0], y.detach().cpu().numpy()[:, 1], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
axs[0, 0].set_title('Gaussian samples')
im2 , _, _, map2 = axs[0, 1].hist2d(sample_flow1.detach().cpu().numpy()[:, 0], sample_flow1.detach().cpu().numpy()[:, 1], range=[[-4, 4], [-4, 4]], bins = nBins)
axs[0, 1].set_title('inverse flow 1')
im3 , _, _, map3 = axs[0, 2].hist2d(sample_flow2.detach().cpu().numpy()[: ,0] ,sample_flow2.detach().cpu().numpy()[: ,1], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
axs[0, 2].set_title('inverse flow 2')
im4 , _, _, map4 = axs[1, 0].hist2d(sample_flow3.detach().cpu().numpy()[:, 0], sample_flow3.detach().cpu().numpy()[:, 1], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
axs[1, 0].set_title('inverse flow 3')
im4 , _, _, map4 = axs[1, 1].hist2d(sample_flow4.detach().cpu().numpy()[:, 0], sample_flow4.detach().cpu().numpy()[:, 1], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
axs[1, 1].set_title('inverse flow 4')
im4 , _, _, map4 = axs[1, 2].hist2d(sample_flow5.detach().cpu().numpy()[:, 0], sample_flow5.detach().cpu().numpy()[:, 1], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
axs[1, 2].set_title('inverse flow 5')

# ----------------------------------------------------------------------------------------------------------
# Plot Generated Samples
# ----------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(1, 1, 1, aspect="equal")
vmin = 0.0; vmax = 1.0

extent = [[LOWX, HIGHX], [LOWY, HIGHY]]
h3, _, _, map3 = ax.hist2d(sample_flow1.detach().cpu().numpy()[:, 0], sample_flow1.detach().cpu().numpy()[:, 1],
          range=extent, bins=nBins)
h3 = h3/(sample_flow1.shape[0])
im3 = ax.imshow(h3)
im3.set_clim(vmin, vmax)
ax.axis('off')
sSaveLoc = os.path.join('./experiments/images/' + args.data + '/' + args.data + '_sample_flow1.png')
plt.savefig(sSaveLoc,bbox_inches='tight')
plt.close(fig)
print("finished plotting to folder", args.save)