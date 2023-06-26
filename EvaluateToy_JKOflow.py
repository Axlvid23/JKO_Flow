import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
import argparse
import datetime
import sys
sys.path.append('./src/utils.py')
sys.path.append('./src/train_subproblem.py')
from src.utils import *
from src.train_subproblem import *
file = './logger.txt'
with open(file, mode='a'): pass

cf = getconfig()

if cf.gpu: # if gpu on platform
    n_samples    = int(6000) #number of data samples
    def_batch    = 2000 #batch size
    def_niter    = 1500 #number of iterations
    n_subproblems  = 5 #number of JKO flow iterations
    max_epochs   = int(1e3) #maximum number of epochs
    def_viz_freq = max_epochs #setting plot frequency to once per subproblem
else:  # if no gpu on platform, assume debugging on a local cpu
    n_samples    = int(6000) #number of data samples
    def_batch    = 2000 #batch size
    def_niter    = 1500 #number of iterations
    n_subproblems  = 5 #number of JKO flow iterations
    max_epochs   = int(1e3) #maximum number of epochs
    def_viz_freq = max_epochs #setting plot frequency to once per subproblem

parser = argparse.ArgumentParser('OT-Flow')
"""#---------------------------- Choose 2d Toy problem ---------------------------#"""
### Select toy problem by choosing default dataset name
### Very important that all of these parameters match those chosen for training in TrainToy_JKOflow.py
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='swissroll'
)

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=8, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0,10.0,0.5')
parser.add_argument('--m'     , type=int, default=16)
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--n_samples'    , type=int  , default=n_samples)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lr'          , type=float, default=1e-2)
parser.add_argument("--drop_freq"   , type=int  , default=100, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=1.0)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='single', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments/cnf/toy')
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

d      = 2
lr     = args.lr
alph   = args.alph
nt     = args.nt
nt_val = args.nt_val
nTh    = args.nTh
m      = args.m
net = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
net = net.to(prec).to(device)

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


#--------------------------------------------------------
# flow all at once:
#--------------------------------------------------------
y = cvt(torch.randn(6000,2))
xk = inf_train_gen(args.data, batch_size=args.n_samples)
xk = cvt(torch.from_numpy(xk))
x = xk
count = n_subproblems
img_directory = './experiments/images/' + args.data + '/'
for i in range(count):
  file_name = './experiments/' + args.data + '_weights/' + args.data + '_weights_iter_' + str(count) + '.pth'
  state = torch.load(file_name,
                       map_location=torch.device(device))
  net.load_state_dict(state['model_state_dict'])
  fx = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
  finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
  genModel = integrate(y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
  save_path = img_directory + 'genModel_' + args.data + '_' + str(count) + '.png'
  makedirs(img_directory)
  save_samples_plot(genModel, save_path)
  count=count-1
  if i==count-1:
    mmd_val = mmd(xk.cpu(), genModel[:,0:2].cpu())
    print('MMD of reconstruction = ', mmd_val)
save_samples_plot(xk, img_directory + args.data + '_image.png')
makedirs(img_directory)
nBins = 80
LOW = -4
HIGH = 4
extent = [[LOW, HIGH], [LOW, HIGH]]

d1 = 0
d2 = 1

# density function of the standard normal
def normpdf(x):
    mu = torch.zeros(1, d, device=x.device, dtype=x.dtype)
    cov = torch.ones(1, d, device=x.device, dtype = x.dtype)  # diagonal of the covariance matrix
    denom = (2 * math.pi) ** (0.5 * d) * torch.sqrt(torch.prod(cov))
    num = torch.exp(-0.5 * torch.sum((x - mu) ** 2 / cov, 1, keepdims=True))
    return num / denom

save_samples_plot(y, img_directory + './gaussian.png')

# ----------------------------------------------------------------------------------------------------------
        # Plot Density
# ----------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(1, 1, 1, aspect="equal")

npts = 100

side = np.linspace(LOW, HIGH, npts)
xx, yy = np.meshgrid(side, side)
x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
with torch.no_grad():
    x = cvt(torch.from_numpy(x))
    nt_val = args.nt
    z = integrate(x, net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
    logqx = z[:, d]
    z = z[:, 0:d]

qz = np.exp(logqx.cpu().numpy()).reshape(npts, npts)
normpdfz = normpdf(z)
rho0 = normpdfz.cpu().numpy().reshape(npts, npts) * qz

im = plt.pcolormesh(xx, yy, rho0)
vmin = np.min(rho0)
vmax = np.max(rho0)
im.set_clim(vmin, vmax)
ax.axis('off')
sSaveLoc = img_directory + args.data + '_density.png'
plt.savefig(sSaveLoc,bbox_inches='tight')
plt.close(fig)