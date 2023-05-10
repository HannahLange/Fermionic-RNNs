import torch
from torch import nn
import numpy as np
import math
import os
import timeit 
import argparse
import gpytorch

from model import Model
from localenergy import get_Eloc, tJ2D_MatrixElements, FH2D_MatrixElements
from helper import save, initialize_torch, parse_input, cost_fct, save_params
import observables as o


# for Gaussian Optimization
from skopt import gp_minimize
import random
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern




# Gaussian optimization: lr, hiddendim, n_samples (per step)

def ramp(x):
    # ---------- initial settings --------------
    device = initialize_torch()
    torch.autograd.set_detect_anomaly(True)

    # --------- Define and initialize the model ------------
    H      = "tJ"
    parser = argparse.ArgumentParser()
    params, density, Nx, Ny, bounds_x, bounds_y, load_model, sz_total, sym, su2_sym, antisym = parse_input(parser, H)
    N_tot  = int(Nx*Ny*density)

    # Define hyperparameters
    batchsize   = 2000
    num_epochs      = 3000
    warmup_steps    = 0
    annealing_steps = int(num_epochs/2)
    max_grad    = None
    hiddendim = 70
    T0 = 0
    n_samples = 200 
    lr_decay = 2000

    #START
    lr, optim = x
    model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, N_target=N_tot, sz_total=sz_total, hidden_dim=hiddendim, weight_sharing=True, device = device)
    model = model.to(device)


    # -------- Optimizer and cost function ------------------
    if optim<0.5:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = gpytorch.optim.NGD(model.parameters(), n_samples, lr=lr)

    # --------- start the training ----------------------
    E_training       = []
    N_training       = []
    S_training       = []
    p_diffs_training = []
    epoch = 1
    while epoch <= num_epochs:
        optimizer.zero_grad() # Clears existing gradients from previous epoch

        # for the cost function
        if T0 != None and epoch > warmup_steps: T = max(T0*(1-(epoch-warmup_steps)/annealing_steps), 0)
        elif T0 != None and epoch < warmup_steps: T = T0
        else: T = None

        if epoch > warmup_steps+annealing_steps:
            optimizer.param_groups[0]['lr'] = lr/np.sqrt(1+(epoch-(warmup_steps+annealing_steps))/lr_decay)

        # loop through batches
        batchsize = np.minimum(batchsize, n_samples)
        try:
            if epoch != 1: 
                try: del batch_samples, samples, log_probs, phases, Eloc, cost, N
                except: pass
            Elocs     = [] 
            samples   = [] 
            Ns = []
            p_diffs = []
            for batch in range(int(n_samples/batchsize)):
                if batch != 0: del c, E, log_probs, phases, batch_samples, N
                batch_samples = model.sample(batchsize)
                N = o.get_particle_no(batch_samples, model, device).detach()
                Ns.append(N.detach())
                samples.append(batch_samples.detach())
                E, c, log_probs, phases, p_diff  = cost_fct(batch_samples, model, device, H, params, bounds_x, bounds_y, N, 0, 0, 0, sz_total, su2_sym, N_tot, sym, T, antisym)
                Elocs.append(E.detach())
                if batch == 0: cost = c
                else: cost += c
            # take care of nan or exploding cost function
            if max_grad != None:
                if torch.abs(cost) > max_grad:
                    cost = cost/torch.abs(cost)*max_grad
            if torch.isnan(cost):
                lr = 0.5*lr
            else:
                cost.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
            optimizer.zero_grad()
            # record samples, E, N etc.
            samples = torch.stack(samples, axis=0)
            samples = torch.reshape(samples, (samples.size()[0]*samples.size()[1], Nx, Ny))
            Elocs = torch.reshape(torch.stack(Elocs, axis = 0),(samples.size()[0],))
            Eloc = Elocs.mean()
            Eloc_var = (Elocs).var(axis=0)
            Ns = torch.reshape(torch.stack(Ns, axis = 0),(samples.size()[0],))
            N = Ns.mean(axis=0)
            N_training.append(N.cpu())
            E_training.append(Eloc.cpu())
            if epoch%10 == 0 or epoch == 1:
                print('Epoch: {}/ {}.............'.format(epoch, num_epochs), end=' ')
                print("Loss: {:.8f}".format(cost.cpu().detach().numpy())+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var))
            epoch += 1
        except RuntimeError:
            print("RuntimeError: The cost function is not updated!")
    return np.real(np.mean(E_training[-20:]))


# define the oprimization routine
bounds = [(100, 3000), (0,1)]
res = gp_minimize(ramp,                       # the function to minimize
                  bounds,                     # the bounds on each dimension of x
                  acq_func="gp_hedge", #EI",  # the acquisition function
                  n_calls=15,                  # the number of evaluations of f
                  n_random_starts=None,          # the number of random initialization points       
                  n_jobs= -1,
                  verbose= True,
                  random_state=12)
print(res)
np.save("opt_res.npy", res["x"])
np.save("func_opt_res.npy", res["fun"])
