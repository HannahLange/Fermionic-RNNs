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
from helper import save, initialize_torch, parse_input, save_params, momentum_cost_fct
import observables as o
from stoch_reconfig import *

# ---------- initial settings --------------
device = initialize_torch()
#torch.autograd.set_detect_anomaly(True)

# --------- Define and initialize the model ------------
H      = "tJ"
parser = argparse.ArgumentParser()
params, density, Nx, Ny, kx, ky, bounds_x, bounds_y, load_model, sz_total, sym_, antisym, hiddendim = parse_input(parser, H)
N_tot  = int(np.round(Nx*Ny*density,0))
print(Nx, Ny, density, N_tot)
# Define hyperparameters
n_samples   = 100
batchsize   = 20
T0          = 0.1
num_epochs      = 2000
warmup_steps    = 0
annealing_steps = int(num_epochs/4)
max_grad    = None
lr          = 0.0001

if bounds_x == bounds_y:
    bounds = bounds_x
else:
    bounds = bounds_x+"_"+bounds_y
fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(params["Jp"]))+"Jz="+str(float(params["Jz"]))+"t="+str(float(params["t"]))+"den="+"{:.2f}".format(density)+"/"
print(fol)
fol_ext = "_h="+str(hiddendim)

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

model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, N_target=N_tot, sz_total=sz_total, hidden_dim=hiddendim, weight_sharing=True, device = device)
model = model.to(device)
model.train()

if load_model:
    print("check if ../"+fol+"model_params"+fol_ext+".pt exists.")
    if os.path.exists("../"+fol+"model_params"+fol_ext+".pt"):
        print("load ../"+fol+"model_params"+fol_ext+".pt")
        model.load_state_dict(torch.load("../"+fol+"model_params"+fol_ext+".pt")) 
        T0 = 0
        fol_ext += "_pretrained"
        fol_ext += "_kx="+str(kx)+"_ky="+str(ky)
    else:
        print("Could not find converged file. Note that its better to converge to the ground state first! Use src/run.py for that!")
else:
    print("Note that its better to converge to the ground state first! Use src/run.py for that!")
    fol_ext += "_kx="+str(kx)+"_ky="+str(ky)
save_params(ml_params, fol, fol_ext)
# -------- Optimizer and cost function ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# --------- start the training ----------------------
E_training       = []
Px_training       = []
Py_training       = []
epoch = 1
countdown = 0
while epoch <= num_epochs:
    start = timeit.default_timer()

    # for the cost function
    if T0 != None and epoch > warmup_steps: T = max(T0*(1-(epoch-warmup_steps)/annealing_steps), 0)
    elif T0 != None and epoch < warmup_steps: T = T0
    else: T = None
    sym = None
    if epoch > warmup_steps+annealing_steps:
        sym = sym_
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr/np.sqrt(1+(epoch-(warmup_steps+annealing_steps))/500)

    # loop through batches
    batchsize = min(batchsize,n_samples)
    samples = []
    for batch in range(int(n_samples/batchsize)):
        batch_samples = model.sample(batchsize)
        samples.append(batch_samples)
    samples = torch.cat(samples, axis=0)
    N = o.get_particle_no(samples, model, device).detach().mean()
    lamb = 10*np.log10(1+9*epoch/500)
    print("Prefactor of momentum constraint:", lamb)
    Elocs = []
    Px = 0
    Py = 0
    cost = 0
    for batch in range(int(n_samples/batchsize)):
        Eloc, c, log_probs, phases, px, py  = momentum_cost_fct(samples[batch*batchsize:(batch+1)*batchsize], model, device, H, params, bounds_x, bounds_y, kx, ky, lamb)
        cost += c
        Elocs.append(Eloc)
        Px += px
        Py += py
        del Eloc, c, px, py, phases, log_probs
    Elocs = torch.cat(Elocs, axis=0)
    Px /= (batch+1)
    Py /= (batch+1)
    cost /= (batch+1)
    optimizer.zero_grad()
    cost.backward() # Does backpropagation and calculates gradients
    optimizer.step()
    Eloc = Elocs.mean()
    Eloc_var = (Elocs).var(axis=0)
    model.zero_grad()
    end = timeit.default_timer()
    E_training.append(Eloc.detach().cpu())
    Px = Px.detach().cpu()
    Py = Py.detach().cpu()
    Px_training.append(Px)
    Py_training.append(Py)
    if ((np.abs(np.real(Px))-kx)**2+(np.abs(np.real(Py))-ky)**2) <1e-4 or countdown!=0:
        countdown +=1
    if countdown > 2000:
        break
    if epoch%10 == 0 or epoch == 1:
        print('Epoch: {}/ {}/ t/epoch={}.............'.format(epoch, num_epochs, round(end-start,2)), end=' ')
        print("Loss: {:.8f}".format(cost)+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var)+", Px: {:.8f}".format(Px)+", Py: {:.8f}".format(Py))
    if epoch == warmup_steps+annealing_steps or epoch == warmup_steps:
        # ----------- save -----------------------------------
        save(model, bounds, fol,fol_ext, n_samples, device)
        np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
        np.save(fol+"/Px"+fol_ext+".npy", np.array(Px_training))
        np.save(fol+"/Py"+fol_ext+".npy", np.array(Py_training))
    epoch += 1
    del samples, N, Elocs, Eloc, Eloc_var, cost
    if sym != None: del sym_samples, sym_log_probs, sym_phases
# ----------- save -----------------------------------
save(model, bounds, fol, fol_ext, n_samples, device)
np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
np.save(fol+"/Px"+fol_ext+".npy", np.array(Px_training))
np.save(fol+"/Py"+fol_ext+".npy", np.array(Py_training))



