import time
from src.utils import *
import torch

"""#---------------------------- Train Phi Function ---------------------------#"""
def train_Phi(optimizer, iter, alph, current_loader, nt, batch_size, lr, nn, max_epoch, weight_decay, outer_iter, tol=1e-6, print_freq=1, plot_freq = 500 ,verbose=True, val_loader = []):
    """
    Train the neural network for JKO Flow.

    :param optimizer:       user defined optimizer
    :param iter:            integer number of iterations
    :param alph:            list of length 3, the alpha value multipliers
    :param current_loader:  current data loader
    :param nt:              integer number of time steps
    :param batch_size:      integer batch size of data to use in the stochastic optimizer
    :param lr:              float learning rate; default = 1e-2 (placeholder for scheduler)
    :param nn:              neural network
    :param max_epoch:       integer maximum number of epochs
    :param weight_decay:    regularization parameter to keep from overfitting; default 1e-4 (placeholder for scheduler)
    :param outer_iter:      integer quantity of JKO flow subproblems
    :param tol:             optimization tolerance
    :param print_freq:      print frequency
    :param plot_freq:       plot frequency

    :return:
        avg_loss                - average loss
        avg_losses              - list of average losses
        avg_costL               - Lagrangian cost
        avg_costC               - Terminal cost
        avg_costR               -
        xk                      - current value of x
        optimizer.state_dict()  - current optimizer state dictionary
        best_params             - best parameters from current iteration
        avg_grad_norm           - average gradient norm
        epoch + 1               - quantity of epochs counted
    """
    prec = torch.float32
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)
    time_meter = AverageMeter()
    h = 1 / nt
    avg_losses = []

    avg_loss = 0.0
    avg_grad_norm = 0.0
    avg_costL = 0.0
    avg_costC = 0.0
    avg_costR = 0.0
    avg_scaling = 0.1
    best_loss = float('inf')

    nn.train()
    for epoch in range(max_epoch):
        start_time = time.time()

        for x_batch, in current_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            loss, costs, xk = compute_loss(iter, nn, x_batch, nt, alph)
            # perform exponential averaging of loss and grad norm values
            costL = costs[0].detach().cpu().numpy()
            costC = costs[1].detach().cpu().numpy()
            costR = costs[2].detach().cpu().numpy()
            avg_loss = avg_scaling * loss.detach().cpu().numpy() + (1 - avg_scaling) * avg_loss
            avg_costL = avg_scaling * costL + (1 - avg_scaling) * avg_costL
            avg_costC = avg_scaling * costC + (1 - avg_scaling) * avg_costC
            avg_costR = avg_scaling * costR + (1 - avg_scaling) * avg_costR

            loss.backward()
            optimizer.step()

        avg_losses.append(avg_loss)

        # ------------------------------------------------------------------------------------------------
        # compute (relative) gradient norms for printing
        # ------------------------------------------------------------------------------------------------
        current_grad_norm = 0.0
        for p in nn.parameters():
            if p.grad == None:
                continue  # some gradients don't exist
            param_norm = torch.norm(p.grad.detach().data)
            current_grad_norm += param_norm.item() ** 2
        current_grad_norm = current_grad_norm ** 0.5
        if epoch == 0:
            total_norm0 = current_grad_norm

        avg_grad_norm = avg_scaling * current_grad_norm / total_norm0 + (1 - avg_scaling) * avg_grad_norm

        end_time = time.time()
        time_epoch = end_time - start_time

        # ------------------------------------------------------------------------------------------------
        # plot
        # ------------------------------------------------------------------------------------------------
        if epoch % plot_freq == 0 or epoch == max_epoch - 1:
            log_message = (
                '{:05d}/{:05d}  {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}  '.format(outer_iter, iter, time_meter.val,
                                                                                       avg_loss, avg_costL, avg_costC,
                                                                                       avg_costR))
            with torch.no_grad():
                nn.eval()
                for xkval, in val_loader:
                    xkval = xkval.to(device)
                    test_loss, test_costs, x_val_temp = compute_loss(iter, nn, xkval, nt=nt, alphas=alph)
                    log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                        test_loss, test_costs[0], test_costs[1], test_costs[2]
                    )
                    # save best set of parameters
                    if test_loss.item() < best_loss:
                        best_loss = test_loss.item()
                        makedirs('experiments/cnf/toy')
                        best_params = nn.state_dict()
                    y = cvt(torch.randn(batch_size, nn.d)).to(device)  # sampling from the standard normal (rho_1)
                    plot4(nn, xkval, y, nt, doPaths=True, sTitle='')
                nn.train()

        if verbose == True and epoch % print_freq == 0:
            print('{:<10s}{:<10s}{:<15s}{:<15s}{:<15s}{:<15s}{:<15s}{:<15s} '.format('iter', 'epoch', 'Loss', 'cost L',
                                                                                     'cost C', 'cost R', 'grad_norm',
                                                                                     'time'))
            print('{:<10d}{:<10d}{:<15f}{:<15f}{:<15f}{:<15f}{:<15f}{:<15f} '.format(outer_iter, epoch, avg_loss,
                                                                                     avg_costL, avg_costC, avg_costR,
                                                                                     avg_grad_norm, time_epoch))
        else:
            None

        if avg_grad_norm <= tol:
            break
        else:
            None
    return avg_loss, avg_losses, avg_costL, avg_costC, avg_costR, xk.detach(), optimizer.state_dict(), best_params, avg_grad_norm, epoch + 1
