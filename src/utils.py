import math
import os
import logging
import torch
import numpy as np
import torch.nn as nn
import copy
from torch.nn.functional import pad
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
from torch.utils.data import Dataset, TensorDataset, DataLoader

def getconfig():
    return ConfigOT()

class ConfigOT:
    """
    gpu - True means GPU available on plaform , False means it's not; this is used for default values
    os  - 'mac' , 'linux'
    """
    gpu = True
    os = 'linux'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.sum = 0  #

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.sum += val
        self.val = val

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stepRK4(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time

    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    z += (1.0/6.0) * K
    return z

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.

    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z

def integrate(x, net, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0], intermediates=False ):
    """
        perform the time integration in the d-dimensional space

    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+3-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """

    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=tspan[0])

    tk = tspan[0]

    if intermediates: # save the intermediate values as well
        zFull = torch.zeros( *z.shape , nt+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
        zFull[:,:,0] = z

        if stepper == 'rk4':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k] , net, alph, tk, tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK1(odefun, zFull[:,:,k] , net, alph, tk, tk+h)
                tk += h

        return zFull

    else:
        if stepper == 'rk4':
            for k in range(nt):
                z = stepRK4(odefun,z,net, alph,tk,tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun,z,net, alph,tk,tk+h)
                tk += h

        return z

    # return in case of error
    return -1

def C(z):
    """Expected negative log-likelihood"""
    d = z.shape[1]-3
    l = z[:,d] # log-det
    return -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d],2) / 2  , 1 , keepdims=True ) + l.unsqueeze(1) )


def odefun(x, t, net, alph=[1.0,1.0,1.0]):
    """
    neural ODE combining the characteristics and log-determinant, the transport costs, and
    the HJB regularizer.

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 3

    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    gradPhi, trH = net.trHess(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    dr = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) 

    return torch.cat( (dx,dl,dv,dr) , 1 )

def sample_rho0(n, mean, var):
  dim = mean.shape[0]
  return var*torch.randn(n, dim, device=device) + mean

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger

def plot4(net, x, y, nt_val, sTitle="", doPaths=False):
    """
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """

    d = net.d
    nSamples = x.shape[0]

    fx = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
    finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
    genModel = integrate(y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)

    invErr = torch.norm(x[:,0:d] - finvfx[:,0:d]) / x.shape[0]

    nBins = 33
    LOWX  = -4
    HIGHX = 4
    LOWY  = -4
    HIGHY = 4

    if d > 50: # assuming bsds
        # plot dimensions d1 vs d2 
        d1=0
        d2=1
        LOWX  = -0.15   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 0.15
        LOWY  = -0.15
        HIGHY = 0.15
    if d > 700: # assuming MNIST
        d1=0
        d2=1
        LOWX  = -10   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 10
        LOWY  = -10
        HIGHY = 10
    elif d==8: # assuming gas
        LOWX  = -2   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX =  2
        LOWY  = -2
        HIGHY =  2
        d1=2
        d2=3
        nBins = 100
    else:
        d1=0
        d2=1

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

    im1 , _, _, map1 = axs[0, 0].hist2d(x.detach().cpu().numpy()[:, d1], x.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
    axs[0, 0].set_title('x from rho_0')
    im2 , _, _, map2 = axs[0, 1].hist2d(fx.detach().cpu().numpy()[:, d1], fx.detach().cpu().numpy()[:, d2], range=[[-4, 4], [-4, 4]], bins = nBins)
    axs[0, 1].set_title('f(x)')
    im3 , _, _, map3 = axs[1, 0].hist2d(finvfx.detach().cpu().numpy()[: ,d1] ,finvfx.detach().cpu().numpy()[: ,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 0].set_title('finv( f(x) )')
    im4 , _, _, map4 = axs[1, 1].hist2d(genModel.detach().cpu().numpy()[:, d1], genModel.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 1].set_title('finv( y from rho1 )')

    fig.colorbar(map1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    fig.colorbar(map2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    fig.colorbar(map3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    fig.colorbar(map4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )


    # plot paths
    if doPaths:
        forwPath = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)
        backPath = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)

        # plot the forward and inverse trajectories of several points; white is forward, red is inverse
        nPts = 10
        pts = np.unique(np.random.randint(nSamples, size=nPts))
        for pt in pts:
            axs[0, 0].plot(forwPath[pt, 0, :].detach().cpu().numpy(), forwPath[pt, 1, :].detach().cpu().numpy(), color='white', linewidth=4)
            axs[0, 0].plot(backPath[pt, 0, :].detach().cpu().numpy(), backPath[pt, 1, :].detach().cpu().numpy(), color='red', linewidth=2)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            # axs[i, j].get_yaxis().set_visible(False)
            # axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')
    plt.show()

def plotAutoEnc(x, xRecreate, sPath):
    # assume square image
    s = int(math.sqrt(x.shape[1]))
    nex = 8
    fig, axs = plt.subplots(4, nex//2)
    fig.set_size_inches(9, 9)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")
    for i in range(nex//2):
        axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
        axs[1, i].imshow(x[ nex//2 + i , : ].reshape(s,s).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[ nex//2 + i , : ].reshape(s, s).detach().cpu().numpy())

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plotAutoEnc3D(x, xRecreate, sPath):
    nex = 8
    fig, axs = plt.subplots(4, nex//2)
    fig.set_size_inches(9, 9)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")
    for i in range(nex//2):
        axs[0, i].imshow(x[i,:].permute(1,2,0).detach().cpu().numpy())
        axs[1, i].imshow(x[ nex//2 + i , : ].permute(1,2,0).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].permute(1,2,0).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[ nex//2 + i , : ].permute(1,2,0).detach().cpu().numpy())


    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()



def plotImageGen(x, xRecreate):

    # assume square image
    s = int(math.sqrt(x.shape[1]))

    nex = 80
    nCols = nex//5


    fig, axs = plt.subplots(7, nCols)
    fig.set_size_inches(16, 7)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nCols):
        axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[nCols + i,:].reshape(s,s).detach().cpu().numpy())
        
        axs[4, i].imshow(xRecreate[2*nCols + i,:].reshape(s,s).detach().cpu().numpy())
        axs[5, i].imshow(xRecreate[3*nCols + i , : ].reshape(s, s).detach().cpu().numpy())
        axs[6, i].imshow(xRecreate[4*nCols + i , : ].reshape(s, s).detach().cpu().numpy())

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()


def plot4mnist(x, sPath, sTitle=""):
    """
    x - tensor (>4, 28,28)
    """
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(sTitle)

    im1 = axs[0, 0].imshow(x[0,:,:].detach().cpu().numpy())
    im2 = axs[0, 1].imshow(x[1,:,:].detach().cpu().numpy())
    im3 = axs[1, 0].imshow(x[2,:,:].detach().cpu().numpy())
    im4 = axs[1, 1].imshow(x[3,:,:].detach().cpu().numpy())

    fig.colorbar(im1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    fig.colorbar(im2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    fig.colorbar(im3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    fig.colorbar(im4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


# Generate Toy Data
# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        return inf_train_gen("8gaussians", rng, batch_size)

#Generate Miniboone Data
def load_data(root_path):

    data = np.load(root_path)
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test

def load_data_normalised(root_path):

    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test

def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**2
        ndecs = 2
    else:
        ndecs += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs

def batch_iter(X, shuffle=True):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    return X[idxs]

# Neural Networks
def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi

        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x

class Phi(nn.Module):
    def __init__(self, nTh, m, d, r=10, alph=[1.0] * 5):
        """
            neural network approximating Phi

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.d    = d
        self.alph = alph

        r = min(r,d+1) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+1) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N = ResNN(d, m, nTh=nTh)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def forward(self, x):
        """ calculating Phi(s, theta)...not used in OT-Flow """

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A

        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)

    def trHess(self,x, justGrad=False ):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh

        N    = self.N
        m    = N.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1]-1
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()

        if justGrad:
            return grad.t()

        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp  = derivTanh(opening.t()) * z[1]
        trH  = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1)) # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()   # act'( K_0 * S + b_0 )
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1,N.nTh):
            KJ  = torch.mm(N.layers[i].weight , Jac.reshape(m,-1) )
            KJ  = KJ.reshape(m,-1,nex)
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            temp = N.layers[i].forward(u[i-1]).t() # (K_i * u_{i-1} + b_i)
            t_i = torch.sum(  ( derivTanh(temp) * term ).reshape(m,-1,nex)  *  torch.pow(KJ,2) ,  dim=(0, 1) )
            trH  = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ # update Jacobian

        return grad.t(), trH + torch.trace(symA[0:d,0:d])
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )

def create_data_loader(old_loader, net, nt):

  x_new = torch.zeros(1)
  with torch.no_grad():
    for x,  in old_loader:
      x_temp = integrate(x, net, [0.0, 1.0], nt, stepper="rk4", alph=net.alph)[:,0:net.d]

      if x_new.shape[0] == 1:
        x_new = x_temp
      else:
        x_new = torch.cat((x_new, x_temp), dim=0)

  new_dataset = TensorDataset(x_new)

  new_data_loader = DataLoader(dataset=new_dataset,
                              batch_size=old_loader.batch_size, shuffle=False)
  
  return new_data_loader



# from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd


def mmd(x,y, indepth=False, alph=1.0):
    """
        from Li et al. Generative Moment Matching Networks 2015
    Gaussian kernel
    :param x: numpy matrix of size (nex, :)
    :param y: numpy matrix of size (nex,:)
    :return: MMD(x,y)
    """
    # convert to numpy
    if type(x) is torch.Tensor:
        x = x.detach().numpy()
    if type(y) is torch.Tensor:
        y = y.detach().numpy()

    if max(x.size,y.size) > 20000:
        indepth = True


    # there's a quick method, that uses a lot of memory, that can be run on pointclouds of a few thousand samples
    # and there's a long and slow way that can be run on pointclouds with 10^5 samples
    if not indepth:
        # make torch tensor
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(torch.float32)
        if type(y) is np.ndarray:
            y = torch.from_numpy(y).to(torch.float32)
        mmdObj = MMDStatistic(x.shape[0],y.shape[0])
        return mmdObj( x , y , [alph] ).item() # just use alpha = 1.0

    else:
        # lots of examples, do a long approach
        # very slow
        # kernel  = exp(  1/(2*sig) * || x - xj ||^2    )
        # sig = 0.5
        # alpha = -1.0 / (2*sig)
        alpha = - alph

        xx  = 0.0
        yy  = 0.0
        xy  = 0.0
        N = x.shape[0]
        M = y.shape[0]

        NsqrTerm  = 1/N**2
        MsqrTerm  = 1/M**2
        crossTerm = -2/(N*M)

        for i in range(N):
            xi = x[i,:]
            diff = xi - x
            power = alpha * np.linalg.norm(diff, ord=2, axis=1, keepdims=True)**2 # nex-by-1
            xx += np.exp(power).sum()
            diff = xi - y
            power = alpha * np.linalg.norm(diff, ord=2, axis=1, keepdims=True) ** 2  # nex-by-1
            xy += np.exp(power).sum()

        for i in range(M):
            yi = y[i,:]
            diff = yi - y
            power = alpha * np.linalg.norm(diff, ord=2, axis=1, keepdims=True)**2 # nex-by-1
            yy += np.exp(power).sum()

        return NsqrTerm*xx + crossTerm*xy + MsqrTerm*yy

def compute_loss(iter, net, x, nt, alphas, method='rk4'): 
    Jc , cs, x = OTFlowProblem(iter, x, net, [0,1], nt=nt, stepper=method, alph=alphas)
    return Jc, cs, x

def OTFlowProblem(iter, x, Phi, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0] ):
    """
    Evaluate objective function of OT Flow problem.

    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        total_cost  - integer, total cost using alphas (alph) provided
        cs          -  list length 5, the five computed costs
        z           - output data tensor nex-by-d
    """
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    if iter==0: 
      z = pad(x, (0, 3, 0, 0), value=0)
      tk = tspan[0]
    else:
      z = x
      tk = tspan[0]

    if stepper=='rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    # ASSUME all examples are equally weighted
    costL  = torch.mean(z[:,-2])
    costC  = torch.mean(C(z))
    costR  = torch.mean(z[:,-1])

    cs = [costL, costC, costR]

    # return dot(cs, alph)  , cs
    # return sum(i[0] * i[1] for i in zip(cs, alph)) , cs, z.detach()
    return alph[0]*cs[0] + alph[1]*cs[1] + alph[2]*cs[2], cs, z.detach()