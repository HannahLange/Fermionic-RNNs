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


# ---------- initial settings --------------
device = initialize_torch()
torch.autograd.set_detect_anomaly(True)

# --------- Define and initialize the model ------------
H      = "tJ"
parser = argparse.ArgumentParser()
params, density, Nx, Ny, bounds_x, bounds_y, load_model, sz_total, sym_, su2_sym, antisym = parse_input(parser, H)
N_tot  = int(Nx*Ny*density)

# Define hyperparameters
n_samples   = 1000
batchsize   = 2000
T0          = 1.0
mu          = max(400 * params["t"], 100)
mu_sz       = 10
mu_sym      = 1
num_epochs      = 30000
warmup_steps    = 1000
annealing_steps = 4000
max_grad    = None
lr = 0.0001
hiddendim   = 150


if bounds_x == bounds_y:
    bounds = bounds_x
else:
    bounds = bounds_x+"_"+bounds_y
fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(params["Jp"]))+"Jz="+str(float(params["Jz"]))+"t="+str(float(params["t"]))+"den="+"{:.2f}".format(density)+"/"
print(fol)
if antisym:
    fol_ext = "_h="+str(hiddendim)
else:
    fol_ext = "_no_antisym"
if load_model:
    T0 = 0
    fol_ext = "2"
    num_epochs = 5000
    annealing_steps  = 1
    warmup_steps = 0
    lr = lr/5
    mu *= 10

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
             "mu": mu, 
             "mu_sz": mu_sz,
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


model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, hidden_dim=hiddendim, weight_sharing=True, device = device)
model = model.to(device)
if load_model:
    if os.path.exists(fol+"model_params.pt"):
        model.load_state_dict(torch.load(fol+"model_params.pt"))    




# -------- Optimizer and cost function ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)

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
    if epoch < warmup_steps+warmup_steps+annealing_steps/2:
        sz_tot = None #sz_tot = sz_total
        mu_sz_log = 0 #mu_sz_log = np.log10(1+9*(epoch-(warmup_steps))/2000)*mu_sz
        mu_sym_log = 0
        sym = None
    if epoch > warmup_steps+annealing_steps/2:
        optimizer.param_groups[0]['lr'] = lr/np.sqrt(1+(epoch-(warmup_steps+annealing_steps/2))/5000)
        sz_tot = sz_total
        mu_sz_log = np.log10(1+9*(epoch-(warmup_steps+annealing_steps/2))/5000)*mu_sz
    if epoch > warmup_steps+annealing_steps:
        sym = sym_
        mu_sym_log = np.log10(1+9*(epoch-(warmup_steps+annealing_steps))/5000)*mu_sym
    mu_log = np.log10(1+9*epoch/20000)*mu
    if load_model: 
        mu_log = 10*mu + np.log10(1+9*epoch/1000)*mu
        mu_sz_log = np.log10(1+9*(epoch)/1000)*mu_sz
        sz_tot = sz_total
        sym = sym_
        mu_sym_log = np.log10(1+9*epoch/5000)*mu_sym
    # loop through batches
    batchsize = np.minimum(batchsize, n_samples)
    #if Nx == Ny: l = int(batchsize/4)
    #else: l = int(batchsize/2)
    if True: #try:
        if epoch != 1: del batch_samples, samples, log_probs, phases, Eloc, cost, N
        Elocs     = [] 
        samples   = [] 
        Ns = []
        p_diffs = []
        for batch in range(int(n_samples/batchsize)):
            if batch != 0: del c, E, log_probs, phases, batch_samples, N
            batch_samples = model.sample(batchsize)
            #batch_samples[2*l:3*l] = torch.rot90(batch_samples[2*l:3*l],2,[1,2])
            #if Nx == Ny:
            #    batch_samples[l:2*l] = torch.rot90(batch_samples[l:2*l],1,[1,2])
            #    batch_samples[3*l:] = torch.rot90(batch_samples[3*l:],3,[1,2])
            #batch_samples = batch_samples[torch.randperm(batch_samples.size()[0])]
            N = o.get_particle_no(batch_samples, model, device).detach()
            Ns.append(N.detach())
            samples.append(batch_samples.detach())
            E, c, log_probs, phases, p_diff  = cost_fct(batch_samples, model, device, H, params, bounds_x, bounds_y, N, mu_log, mu_sz_log, mu_sym_log, sz_tot, su2_sym, N_tot, sym, T, antisym)
            Elocs.append(E.detach())
            if batch == 0: cost = c
            else: cost += c
            if p_diff != None: p_diffs.append(p_diff)
        # take care of nan or exploding cost function
        if max_grad != None:
            if torch.abs(cost) > max_grad:
                cost = cost/torch.abs(cost)*max_grad
        if torch.isnan(cost):
            lr = 0.5*lr
        else:
            cost.backward() # Does backpropagation and calculates gradients
    #except RuntimeError:
    #    print("RuntimeError: The cost function is not updated!")
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
    eres = torch.exp(torch.log(Elocs+33.4823).mean())/(Nx*Ny)
    if sym != None:
        p_diffs = torch.reshape(torch.stack(p_diffs, axis = 0),(samples.size()[0],)).mean()
        p_diffs_training.append(p_diffs.cpu())
    end = timeit.default_timer()
    N_training.append(N.cpu())
    E_training.append(Eloc.cpu())
    if epoch%10 == 0 or epoch == 1:
        sx = torch.mean(torch.abs(o.get_sx(batch_samples, log_probs, phases, model, device)))
        sy = torch.mean(torch.abs(o.get_sy(batch_samples, log_probs, phases, model, device)))
        sz = torch.mean(torch.abs(o.get_sz(batch_samples, model, device)))
        S_training.append([sx.detach().cpu(),sy.detach().cpu(),sz.detach().cpu()])
        print('Epoch: {}/ {}/ t/epoch={}.............'.format(epoch, num_epochs, round(end-start,2)), end=' ')
        print("Loss: {:.8f}".format(cost.cpu().detach().numpy())+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var)+", E_res/N: {:.8f}".format(eres), end=" ")
        if sym != None: print(", Deltap_sym: {:.8f}".format(p_diffs), end=" ")
        print(", Sx: {:.4f}".format(sx)+", Sy: {:.4f}".format(sy)+", Sz: {:.4f}".format(sz)+", N/N_tot: {:.4f} %".format(N.cpu().numpy()/N_tot))
        del sx, sy, sz
    if epoch == warmup_steps+annealing_steps:
        # ----------- save -----------------------------------
        save(model, bounds, fol,fol_ext, n_samples, device)
        np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
        np.save(fol+"/N"+fol_ext+".npy", np.array(N_training))
        np.save(fol+"/S"+fol_ext+".npy", np.array(S_training))
        if sym != None: np.save(fol+"/p_sym"+fol_ext+".npy", np.array(p_diffs_training))
    epoch += 1
# ----------- save -----------------------------------
save(model, bounds, fol, fol_ext, n_samples, device)
np.save(fol+"/Eloc"+fol_ext+".npy", np.array(E_training))
np.save(fol+"/N"+fol_ext+".npy", np.array(N_training))
np.save(fol+"/S"+fol_ext+".npy", np.array(S_training))
if sym != None: np.save(fol+"/p_sym"+fol_ext+".npy", np.array(p_diffs_training))



 
