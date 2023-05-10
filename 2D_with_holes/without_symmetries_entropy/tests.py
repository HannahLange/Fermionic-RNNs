import torch
from torch import nn
import numpy as np
import os
import argparse

from model import Model
from localenergy import get_Eloc, tJ2D_MatrixElements, FH2D_MatrixElements
from helper import save, initialize_torch, parse_input, cost_fct, save_params
import observables as o


# ---------- initial settings --------------
device = initialize_torch()

# --------- Define and initialize the model ------------
H      = "tJ"
parser = argparse.ArgumentParser()
params, density, Nx, Ny, bounds_x, bounds_y, load_model, sz_total, sym_, su2_sym, antisym = parse_input(parser, H)
N_tot  = int(Nx*Ny*density)

# Define hyperparameters
n_samples   = 1000
batchsize   = 2000
T0          = 0.0
mu          = max(400 * params["t"], 400)
mu_sz       = 10
mu_sym      = 1
num_epochs      = 20000
warmup_steps    = 0
annealing_steps = 5000
max_grad    = None
lr = 0.0001
hiddendim   = 100


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


model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, hidden_dim=hiddendim, weight_sharing=True, device = device)
model = model.to(device)
model.load_state_dict(torch.load(fol+"model_params"+fol_ext+".pt"))  
model.eval()

samples = model.sample(1000)
log_probs, phases = model.log_probabilities(samples)

# For analyzing the spatial symmetry
symmetric_samples = torch.zeros((samples.size()[0]*3,samples.size()[1], samples.size()[2]), dtype = torch.int32)
l = int(samples.size()[0])
symmetric_samples[:l]    = torch.rot90(samples,    1, [1,2]).cpu().detach()
symmetric_samples[l:2*l] = torch.rot90(samples, 2, [1,2]).cpu().detach()
symmetric_samples[2*l:]  = torch.rot90(samples,  3, [1,2]).cpu().detach()
sym_log_probs, sym_phases = model.log_probabilities(symmetric_samples.long())

s = samples.detach().cpu().numpy()
np.save(fol+"samples"+fol_ext+".npy", s)
probs = np.array([log_probs.cpu().detach().numpy(), sym_log_probs[:l].cpu().detach().numpy(), sym_log_probs[l:2*l].cpu().detach().numpy(), sym_log_probs[2*l:].cpu().detach().numpy()])
np.save(fol+"symmetry"+fol_ext+".npy", probs)


#To analyze the fermionic antisymmetry
Jp = params["Jp"]
Jz = params["Jz"]
t  = params["t"]

Nx         = samples.size()[1]
Ny         = samples.size()[2]
numsamples = samples.size()[0]
if bounds_x == "periodic":
    length_x = Nx
elif bounds_x == "open":
    length_x = Nx-1
if bounds_y == "periodic":
    length_y = Ny
elif bounds_y == "open":
    length_y = Ny-1

_,_, xprime, signs = tJ2D_MatrixElements(Jp, Jz, t, samples, length_x, length_y, True, device)
signs = torch.reshape(signs, (signs.size()[0]*signs.size()[1],))
xprime = torch.reshape(xprime[int(xprime.size()[0]/2):], (int(xprime.size()[0]/2)*xprime.size()[1], xprime.size()[2], xprime.size()[3])).to(torch.int32)
_, antisym_phases = model.log_probabilities(xprime.to(torch.int64)) 

np.save(fol+"antisymmetry_signs"+fol_ext+".npy", signs.detach().cpu().numpy())
np.save(fol+"antisymmetry_phases"+fol_ext+".npy", phases.detach().cpu().numpy())
np.save(fol+"antisymmetry_xprime_phases"+fol_ext+".npy", antisym_phases.detach().cpu().numpy())
