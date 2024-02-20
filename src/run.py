import torch
print(torch.__version__)
from torch import nn
import numpy as np
import numpy.random as random
import math
import os
import timeit 
import argparse

from model import Model
from localenergy import get_Eloc, tJ2D_MatrixElements
from helper import save, initialize_torch, parse_input, cost_fct, save_params
import observables as o
from stoch_reconfig import *

# ---------- initial settings --------------
device = initialize_torch()

# --------- Define and initialize the model ------------
H      = "tJ"
parser = argparse.ArgumentParser()
params, density, Nx, Ny, bounds_x, bounds_y, load_model, sz_total, sym, antisym, hiddendim = parse_input(parser, H)
N_tot  = int(Nx*Ny*density)

# --------- Define hyperparameters ---------
n_samples   = 400        # number of samples used in each VMC step
batchsize   = 400        # batch size
T0          = 0          # initial temperature for variational annealing, if T0=0 no annealing is used
num_epochs  = 20000      # total number of epochs
warmup_steps= 0          # warmup steps: if epoch<warmup_steps learning rate and temperature for annealing do not decrease
max_grad    = None       # cut off gradents if set to a float 
lr          = 0.0001     # initial learning rate
minSR       = False      # use special run_sr.py script

annealing_steps = int(num_epochs/4) #annealing of learning rate (not T!), if minSR, then also the diagonal offset is kept constant during these steps, before being decreased afterwards



# --------- prepare everything for the run ---------
if bounds_x == bounds_y:
    bounds = bounds_x
else:
    bounds = bounds_x+"_"+bounds_y


fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(params["Jp"]))+"Jz="+str(float(params["Jz"]))+"t="+str(float(params["t"]))+"den="+"{:.2f}".format(density)+"/"
print(fol)
fol_ext = "_h="+str(hiddendim)
if not antisym:
    fol_ext += "_no_antisym"

if sz_total == None:
    sz_str = "-"
else:
    sz_str = str(sz_total)
if T0 == None:
    T0_str = "-"
else:
    T0_str = str(T0)
ml_params = {"n_samples": n_samples, 
             "warmup_steps": warmup_steps, 
             "annealing_steps": annealing_steps, 
             "num_epochs": num_epochs, 
             "max_grad": max_grad, 
             "learning rate": lr, 
             "hidden dimension": hiddendim, 
             "sz total: ": sz_str,
             "T0": T0_str}
save_params(ml_params, fol, fol_ext)

# --------- set up the model ---------
model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, N_target=N_tot, sz_total=sz_total, hidden_dim=hiddendim, weight_sharing=True, device = device)
model = model.to(device)
model.train()

if load_model:
    print("check if "+fol+"model_params"+fol_ext+".pt exists.")
    if os.path.exists(fol+"model_params"+fol_ext+".pt"):
        print("load "+fol+"model_params"+fol_ext+".pt")
        model.load_state_dict(torch.load(fol+"model_params"+fol_ext+".pt")) 
        fol_ext += "_2"


# -------- Optimizer and cost function ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# --------- start the training ----------------------
E_training       = []
N_training       = []
S_training       = []
epoch = 1
while epoch <= num_epochs:
    start = timeit.default_timer()

    # for the variational annealing
    if T0 != None and epoch > warmup_steps: T = max(T0*(1-(epoch-warmup_steps)/1000), 0)
    elif T0 != None and epoch < warmup_steps: T = T0
    else: T = None
    #decrease learning rate with 1/sqrt(1+epoch/500)
    if epoch > warmup_steps+annealing_steps: 
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr/np.sqrt(1+(epoch-(warmup_steps+annealing_steps))/500)

    batchsize = min(batchsize,n_samples)
    samples = []
    for batch in range(int(n_samples/batchsize)):
        batch_samples = model.sample(batchsize)
        samples.append(batch_samples)
        del batch_samples
    samples = torch.cat(samples, axis=0)
    N = o.get_particle_no(samples, model, device).detach().mean()
    if epoch < warmup_steps+2 or not minSR:
        cost = 0
        Elocs = []
        for batch in range(int(n_samples/batchsize)):
            Eloc, c, log_probs, phases  = cost_fct(samples[batch*batchsize:(batch+1)*batchsize], model, device, H, params, bounds_x, bounds_y, sym, T, antisym)
            cost += c
            Elocs.append(Eloc)
        Elocs = torch.cat(Elocs, axis=0)
        cost /= (batch+1)
        optimizer.zero_grad()
        cost.backward() # Does backpropagation and calculates gradients
        optimizer.step()
        optimizer.zero_grad()
    else:
        raise NotImplementedError("use run_sr.py")
    Eloc = Elocs.mean()
    Eloc_var = (Elocs).var(axis=0)
    end = timeit.default_timer()
    N_training.append(N.detach().cpu())
    E_training.append(Eloc.detach().cpu())
    if epoch%10 == 0 or epoch == 1:
        print('Epoch: {}/ {}/ t/epoch={}.............'.format(epoch, num_epochs, round(end-start,2)), end=' ')
        print("Loss: {:.8f}".format(cost)+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var))
    if epoch == warmup_steps+annealing_steps or epoch == warmup_steps: #save intermediate steps
        save(model, bounds, fol,fol_ext, n_samples, device)
        np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
        np.save(fol+"/N"+fol_ext+".npy", np.array(N_training))
        np.save(fol+"/S"+fol_ext+".npy", np.array(S_training))
    epoch += 1
    del samples, log_probs, phases, N, Elocs, Eloc, Eloc_var, cost
    if sym != None: del sym_samples, sym_log_probs, sym_phases


# ----------- save -----------------------------------
save(model, bounds, fol, fol_ext, n_samples, device)
np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
np.save(fol+"/N"+fol_ext+".npy", np.array(N_training))
np.save(fol+"/S"+fol_ext+".npy", np.array(S_training))



