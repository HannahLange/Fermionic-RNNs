import torch
from torch import nn
import numpy as np
import numpy.random as random
import math
import os
import timeit 
import argparse

from model import Model
from localenergy import get_Eloc, tJ2D_MatrixElements, FH2D_MatrixElements
from helper import save, initialize_torch, parse_input, cost_fct, save_params
import observables as o
from stoch_reconfig import *

# ---------- initial settings --------------
device = initialize_torch()
torch.autograd.set_detect_anomaly(True)

# --------- Define and initialize the model ------------
H      = "tJ"
parser = argparse.ArgumentParser()
params, density, Nx, Ny, bounds_x, bounds_y, load_model, sz_total, sym_, su2_sym, antisym = parse_input(parser, H)
N_tot  = int(Nx*Ny*density)

# Define hyperparameters
n_samples       = 2000
T0              = 0.0
mu_sym          = 1
num_epochs      = 20000
warmup_steps    = 500
annealing_steps = int(num_epochs/4)
max_grad        = None
lr              = 0.01
hiddendim       = 50


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
             "batchsize": batchsize, 
             "mu_sym": mu_sym,
             "warmup_steps": warmup_steps, 
             "annealing_steps": annealing_steps, 
             "num_epochs": num_epochs, 
             "max_grad": max_grad, 
             "learning rate": lr, 
             "hidden dimension": hiddendim, 
             "sz total: ": sz_str,
             "T0": T0_str}
save_params(ml_params, fol, fol_ext)


model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, N_target=N_tot, sz_total=sz_total, hidden_dim=hiddendim, weight_sharing=True, device = device)
model = model.to(device)
if load_model:
    print("check if "+fol+"model_params"+fol_ext+".pt exists.")
    if os.path.exists(fol+"model_params"+fol_ext+".pt"):
        print("load "+fol+"model_params"+fol_ext+".pt")
        model.load_state_dict(torch.load(fol+"model_params"+fol_ext+".pt")) 
        T0 = 0
        fol_ext = "2"
        num_epochs = 10000
        annealing_steps  = 1000
        warmup_steps = 0
        lr = lr/1000



# -------- Optimizer and cost function ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# --------- start the training ----------------------
E_training       = []
N_training       = []
S_training       = []
p_diffs_training = []
epoch = 1
while epoch <= num_epochs:
    start = timeit.default_timer()
    optimizer.zero_grad() # Clears existing gradients from previous epoch

    # for the cost function
    if T0 != None and epoch > warmup_steps: T = max(T0*(1-(epoch-warmup_steps)/annealing_steps), 0)
    elif T0 != None and epoch < warmup_steps: T = T0
    else: T = None
    mu_sym_log = mu_sym
    sym = sym_
    if epoch > warmup_steps+annealing_steps:
        sym = sym_
        mu_sym_log = np.log10(1+9*(epoch-(warmup_steps+annealing_steps))/5000)*mu_sym
        #optimizer.param_groups[0]['lr'] = lr/np.sqrt(1+(epoch-(warmup_steps+annealing_steps))/2000)

    # loop through batches
    if True: #try:
        if epoch != 1: del batch_samples, samples, log_probs, phases, Eloc, cost, N
        Elocs     = [] 
        samples   = [] 
        Ns = []
        p_diffs = []
        samples = model.sample(n_samples)
        Ns = o.get_particle_no(samples, model, device).detach()
        Elocs, cost, log_probs, phases, p_diffs  = cost_fct(batch_samples, model, device, H, params, bounds_x, bounds_y, mu_sym_log, sym, T, antisym)
        # take care of nan or exploding cost function
        if max_grad != None:
            if torch.abs(cost) > max_grad:
                cost = cost/torch.abs(cost)*max_grad
        cost.backward() # Does backpropagation and calculates gradients
    #except RuntimeError:
    #    print("RuntimeError: The cost function is not updated!")
    run_sr(model,Elocs,samples,optimizer)
    optimizer.zero_grad()
    # record samples, E, N etc.
    Eloc = Elocs.mean()
    Eloc_var = (Elocs).var(axis=0)
    N = Ns.mean(axis=0)
    if p_diffs != None:
        p_diffs_training.append(p_diffs.cpu())
    end = timeit.default_timer()
    N_training.append(N.detach().cpu())
    E_training.append(Eloc.detach().cpu())
    if epoch%10 == 0 or epoch == 1:
        print('Epoch: {}/ {}/ t/epoch={}.............'.format(epoch, num_epochs, round(end-start,2)), end=' ')
        print("Loss: {:.8f}".format(cost.cpu().detach().numpy())+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var))
        if p_diff != None: print(", Deltap_sym: {:.8f}".format(p_diffs))
    if epoch == warmup_steps+annealing_steps:
        # ----------- save -----------------------------------
        save(model, bounds, fol,fol_ext, n_samples, device)
        np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
        np.save(fol+"/N"+fol_ext+".npy", np.array(N_training))
        np.save(fol+"/S"+fol_ext+".npy", np.array(S_training))
        if p_diff != None: np.save(fol+"/p_sym"+fol_ext+".npy", np.array(p_diffs_training))
    epoch += 1
# ----------- save -----------------------------------
save(model, bounds, fol, fol_ext, n_samples, device)
np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
np.save(fol+"/N"+fol_ext+".npy", np.array(N_training))
np.save(fol+"/S"+fol_ext+".npy", np.array(S_training))
if p_diff != None: np.save(fol+"/p_sym"+fol_ext+".npy", np.array(p_diffs_training))



 
