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
from localenergy import get_Eloc, tJ2D_MatrixElements, FH2D_MatrixElements
from helper import save, initialize_torch, parse_input, cost_fct, save_params
import observables as o
from stoch_reconfig import *

# ---------- initial settings --------------
device = initialize_torch()
#torch.autograd.set_detect_anomaly(True)

# --------- Define and initialize the model ------------
H      = "tJ"
parser = argparse.ArgumentParser()
params, density, Nx, Ny, bounds_x, bounds_y, load_model, sz_total, sym_, antisym, hiddendim = parse_input(parser, H)
N_tot  = int(Nx*Ny*density)

# Define hyperparameters
n_samples   = 5000
batchsize   = 500

if bounds_x == bounds_y:
    bounds = bounds_x
else:
    bounds = bounds_x+"_"+bounds_y
fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(params["Jp"]))+"Jz="+str(float(params["Jz"]))+"t="+str(float(params["t"]))+"den="+"{:.2f}".format(density)+"/"
print(fol)
fol_ext = "_h="+str(hiddendim)
if not antisym:
    fol_ext += "_no_antisym"

model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, N_target=N_tot, sz_total=sz_total, hidden_dim=hiddendim, weight_sharing=True, device = device)
model = model.to(device)
model.train()

model.load_state_dict(torch.load(fol+"model_params"+fol_ext+"_2.pt")) 

fol_ext += "_final"


batchsize = min(batchsize,n_samples)
Emean = 0
Evar = 0
n = 0
for batch in range(int(n_samples/batchsize)):
    samples = model.sample(batchsize)
    Eloc, log_probs, phases = get_Eloc(H,params,samples, model, bounds_x, bounds_y, None, antisym)
    Eloc = Eloc.detach().cpu()
    Emean += torch.mean(Eloc)
    Evar += torch.var(Eloc)
    n+=1
    print(n)
    del samples, Eloc
E = [1/n*Emean.numpy(), np.sqrt(1/(n**2)*Evar.numpy())]
print(E)
np.save(fol+"/Eloc"+fol_ext+".npy", E)



