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

# Define hyperparameters, same as in run.py
n_samples   = 1000
batchsize   = 2000
T0          = 0.0
num_epochs      = 20000
warmup_steps    = 1
annealing_steps = 1000 
max_grad    = None
lr          = 0.001
minSR       = True
sym = None
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


# -------- Define model --------
model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, N_target=N_tot, sz_total=sz_total, hidden_dim=hiddendim, weight_sharing=True, device = device)
model = model.to(device)
model.train()

if load_model:
    print("check if "+fol+"model_params"+fol_ext+".pt exists.")
    if os.path.exists(fol+"model_params"+fol_ext+".pt"):
        print("load "+fol+"model_params"+fol_ext+".pt")
        model.load_state_dict(torch.load(fol+"model_params"+fol_ext+".pt")) 
        T0 = 0
        fol_ext = "2"


# -------- Optimizer and cost function ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# --------- start the training ----------------------
E_training       = []
N_training       = []
S_training       = []
epoch = 1
while epoch <= num_epochs:
    start = timeit.default_timer()

    # for the cost function
    if T0 != None and epoch > warmup_steps: T = max(T0*(1-(epoch-warmup_steps)/annealing_steps), 0)
    elif T0 != None and epoch < warmup_steps: T = T0
    else: T = None
    if epoch > warmup_steps+annealing_steps:
        optimizer.param_groups[0]['lr'] = lr/np.sqrt(1+(epoch-(warmup_steps+annealing_steps))/500)
    
    # loop through batches
    batchsize = min(batchsize,n_samples)
    samples = []
    for batch in range(int(n_samples/batchsize)):
        batch_samples = model.sample(batchsize)
        samples.append(batch_samples)
    samples = torch.cat(samples, axis=0)
    N = o.get_particle_no(samples, model, device).detach().mean()
    if minSR and epoch == warmup_steps+1:
        print("Start Stochastic Reconfiguration.")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if epoch < warmup_steps+1 or not minSR:
        Elocs, cost, log_probs, phases  = cost_fct(samples, model, device, H, params, bounds_x, bounds_y, sym, T, antisym)
        optimizer.zero_grad()
        cost.backward(retain_graph=True) # Does backpropagation and calculates gradients
        optimizer.step()
        #optimizer.zero_grad()
        inversion_error = None
        tdvp_error = None
    else:
        if sym != None:
            Elocs, log_probs, phases, sym_samples, sym_log_probs, sym_phases = get_Eloc(H,params,samples, model, bounds_x, bounds_y, sym, antisym)
            sym_log_psi = (0.5*sym_log_probs+1j*sym_phases)
        else:
            Elocs, log_probs, phases = get_Eloc(H,params,samples, model, bounds_x, bounds_y, sym, antisym)
        cost = Elocs.mean()
        if epoch == 1:
            cost.backward(retain_graph=True)
        if epoch <= annealing_steps: diag_offset = 1
        else: diag_offset = min(1,max(1/(np.exp((epoch-annealing_steps)/1000)),1e-10))
        if epoch % 10==0: print("diag_offset: ", diag_offset, ", learning rate: ", optimizer.param_groups[0]['lr'])
        cost, inversion_error, tdvp_error,_,_ = run_sr(model,Elocs,samples,optimizer,diag_offset) # Updates the weights accordingly
    Eloc = Elocs.mean()
    Eloc_var = (Elocs).var(axis=0)
    end = timeit.default_timer()
    N_training.append(N.detach().cpu())
    E_training.append(Eloc.detach().cpu())
    if epoch%10 == 0 or epoch == 1:
        print('Epoch: {}/ {}/ t/epoch={}.............'.format(epoch, num_epochs, round(end-start,2)), end=' ')
        print("Loss: {:.8f}".format(cost)+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var))
        if inversion_error != None: print("    inversion error:"+str(inversion_error.detach().cpu().numpy())+", tdvp error:"+str(tdvp_error.detach().cpu().numpy()))
    if epoch == warmup_steps+annealing_steps or epoch == warmup_steps:
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



