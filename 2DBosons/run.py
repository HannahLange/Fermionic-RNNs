import torch
from torch import nn
import numpy as np
import numpy.random as random
import math
import os
import timeit 

from model import Model
from localenergy import XXZ2D_Eloc
import observables as o

def save(model, boundaries, folder, device):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder+"model_params.pt")
    # calculate the nearest neighbor spin correlators
    samples, log_probs, phases = model.sample(1000)
    szsz = np.array(o.get_szsz(samples, log_probs, boundaries, model, device))
    np.save(folder+"szsz.npy", szsz)
    sxsx = np.array(o.get_sxsx(samples, log_probs, phases, boundaries, model, device))
    np.save(folder+"sxsx.npy", sxsx)
    sysy = np.array(o.get_sysy(samples, log_probs, phases, boundaries, model, device))
    np.save(folder+"sysy.npy", sysy)
    
def cost_fct(samples, model, Jp1, Jz1, Jp2, Jz2, boundaries, sz_tot=None, symmetry=None):
    if symmetry != None:
        Eloc, log_probs, phases, sym_samples, sym_log_probs, sym_phases = XXZ2D_Eloc(Jp1, Jz1, Jp2, Jz2, 
samples, model, boundaries, symmetry)
        sym_log_psi = (0.5*sym_log_probs+1j*sym_phases)
    else:
        Eloc, log_probs, phases = XXZ2D_Eloc(Jp1, Jz1, Jp2, Jz2, samples, model, boundaries, symmetry)
    log_psi = (0.5*log_probs+1j*phases)
    eloc_sum = (Eloc).mean(axis=0)
    e_loc_corr = (Eloc - eloc_sum).detach()
    if sz_tot != None:
        e_loc_corr += (o.get_sz_(samples, model, device).detach()-sz_tot*torch.ones((samples.size()[0])))**2 
    cost = 2 * torch.real((torch.conj(log_psi) * e_loc_corr.to(torch.complex128))).mean(axis=0)
    if symmetry != None:
        cost += 2 * torch.abs(torch.real((torch.conj(log_psi)-(torch.conj(sym_log_psi))))).mean(axis=0)
    return Eloc, cost, log_probs, phases



# ---------- initial settings --------------
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
random.seed(1234)
np.random.seed(0)
torch.manual_seed(1234)


# --------- Define and initialize the model ------------

# Define model parameters
Jp1        = 1
Jz1        = 1
Jp2        = 0
Jz2        = 0
Nx         = 8
Ny         = 8
bounds     = "open"
load_model = False

# Define hyperparameters
n_samples   = 200
sym         = None
n_epochs    = 1000
lr          = 0.01
lr_decay    = 200
lr_thresh   = 0.0005
hiddendim   = min(int(Nx*Ny/2),20)

if Jp2 == 0 and Jz2 == 0:
    fol = str(Nx)+"x"+str(Ny)+"_qubits/Jp="+str(float(Jp1))+"Jz="+str(float(Jz1))+"/"
else:
    fol = str(Nx)+"x"+str(Ny)+"_qubits/Jp="+str(float(Jp1))+"Jz="+str(float(Jz1))+"Jp_nn="+str(float(Jp2))+"Jz_nn="+str(float(Jz2))+"/"
model = Model(input_size=2, system_size_x=Nx,system_size_y=Ny, hidden_dim=hiddendim, n_layers=1, device = device)
model = model.to(device)
model = model.double()
if load_model:
    if os.path.exists(fol+"model_params.pt"):
        model.load_state_dict(torch.load(fol+"model_params.pt"))    

# -------- Optimizer and cost function ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
# --------- start the training ----------------------
Elocs = []
for epoch in range(1, n_epochs + 1):
    start = timeit.default_timer()
    optimizer.param_groups[0]['lr'] = max(lr_thresh, lr/(1+epoch/lr_decay))
    samples = model.sample(n_samples)
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    Eloc, cost, log_probs, phases = cost_fct(samples, model, Jp1, Jz1, Jp2, Jz2, bounds, sz_tot=0, symmetry=sym)
    cost.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    optimizer.zero_grad()
    sx = o.get_sx(samples, log_probs, phases, model, device)
    sy = o.get_sy(samples, log_probs, phases, model, device)
    sz = o.get_sz(samples, model, device)
    if (epoch >= 1/3*n_epochs) and sym == None:
        print("Use spatial symmetry from now.")
        if Jp2 == 0 and Jz2 == 0: #square lattice 
            sym = "C4"
        else:
            sym = "C2"
    Elocs = (Eloc).mean(axis=0)
    end = timeit.default_timer()
    if epoch%10 == 0 or epoch == 1:
        print('Epoch: {}/ {}/ t/epoch={}.............'.format(epoch, n_epochs, round(end-start,2)), 
end=' ')
        print("Loss: {:.8f}".format(cost)+", mean(E): {:.8f}".format((Eloc).mean(axis=0))+", var(E): {:.8f}".format((Eloc).var(axis=0))+", Sx: {:.4f}".format(sx)+", Sy: {:.4f}".format(sy)+", Sz: {:.4f}".format(sz))
    del cost, samples
# ----------- save -----------------------------------
save(model, bounds, fol, device)
np.save(fol+"/Eloc.npy", np.array(Elocs.detach().numpy()))

