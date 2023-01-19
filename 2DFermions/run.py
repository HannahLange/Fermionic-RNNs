import torch
from torch import nn
import numpy as np
import numpy.random as random
import math
import os
import timeit 

from model import Model
from localenergy import tJ2D_Eloc, tJ2D_MatrixElements
import observables as o

def save(model, boundaries, folder, device):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder+"model_params.pt")
    # calculate the nearest neighbor spin correlators
    samples = model.sample(1000)
    log_probs, phases = model.log_probabilities(samples)
    szsz = np.array(o.get_szsz(samples, log_probs, boundaries, model, device))
    np.save(folder+"szsz.npy", szsz)
    sxsx = np.array(o.get_sxsx(samples, log_probs, phases, boundaries, model, device))
    np.save(folder+"sxsx.npy", sxsx)
    sysy = np.array(o.get_sysy(samples, log_probs, phases, boundaries, model, device))
    np.save(folder+"sysy.npy", sysy)

    
def cost_fct(samples, model, Jp, Jz, t, boundaries, lamb, N, mu, sz_tot, N_tot, beta, symmetry):
    cost = 0
    if symmetry != None:
        Eloc, log_probs, phases, sym_samples, sym_log_probs, sym_phases = tJ2D_Eloc(Jp, Jz, t,samples, model, boundaries, symmetry)
        sym_log_psi = (0.5*sym_log_probs+1j*sym_phases)
    else:
        Eloc, log_probs, phases = tJ2D_Eloc(Jp, Jz, t, samples, model, boundaries, symmetry)
    log_psi = (0.5*log_probs+1j*phases)

    eloc_sum = (Eloc).mean(axis=0)
    e_loc_corr = mu*(1-lamb)*Eloc
    e_loc_corr += mu*lamb*(Eloc - eloc_sum)
    if sz_tot != None:
        e_loc_corr += (o.get_sz_(samples, model, device)-sz_tot*torch.ones((samples.size()[0])))**2     
    if N_tot != None:
        e_loc_corr += (N/N_tot-torch.ones((samples.size()[0])))**2 
    if symmetry != None:
        cost += 4 * ((torch.exp(log_probs)-torch.exp(sym_log_probs)) * (torch.real((torch.conj(log_psi)-(torch.conj(sym_log_psi)))))).mean(axis=0)
        #e_loc_corr += (torch.exp(log_probs) - torch.exp(sym_log_probs))**2
    cost += 2 * torch.real((torch.conj(log_psi) * e_loc_corr.detach().to(torch.complex128))).mean(axis=0)
    print(cost)
    beta_term = beta*(torch.norm(model.rnn.W1, p=1)+torch.norm(model.rnn.W2, p=1)+torch.norm(model.rnn.W3, p=1))
    cost += beta_term
    print(beta_term)
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
Jz         = 1.0
Jp         = 0.75
t          = 0.0
density    = 1-1/16
Nx         = 4
Ny         = 4
bounds     = "open"
load_model = True
sz_tot     = None #0.5
N_tot      = int(Nx*Ny*density)

# Define hyperparameters
n_samples   = 200
mu          = 0.01
lamb        = 1.0
beta        = 0.0
sym         = None
n_epochs    = 1000
max_grad    = 1000
lr          = 0.1
lr_decay    = 1000
lr_thresh   = 0.0005
hiddendim   = int(N_tot/2)

fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(Jp))+"Jz="+str(float(Jz))+"t="+str(float(t))+"den="+"{:.2f}".format(density)+"/"
print(fol)
model = Model(input_size=3, system_size_x=Nx,system_size_y=Ny, hidden_dim=hiddendim, n_layers=1, device = device)
model = model.to(device)
model = model.double()
if load_model:
    if os.path.exists(fol+"model_params.pt"):
        model.load_state_dict(torch.load(fol+"model_params.pt"))    




# -------- Optimizer and cost function ------------------
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)

# --------- start the training ----------------------
n = 0
for epoch in range(1, n_epochs + 1):
    start = timeit.default_timer()
    optimizer.param_groups[0]['lr'] = max(lr_thresh, lr/(1+n/lr_decay))
    samples = model.sample(n_samples)
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    N  = o.get_particle_no(samples, model, device)
    if torch.abs(N.mean()/N_tot-1)<=0.0005:
        if beta == 0:
            beta = 0.01
            print("Set beta to "+str(beta))
        if beta != 0:
            n += 1
        if beta != 0 and mu < 1 and n >=20:
            #sym = None #"C4"
            mu = 1
            #lr = lr/10
            beta = 0.01
            print("Set mu to "+str(mu))
    Elocs, cost, log_probs, phases = cost_fct(samples, model, Jp, Jz, t, bounds, lamb, N, mu, sz_tot, N_tot, beta, symmetry=sym)
    if max_grad != None:
        if torch.abs(cost) > max_grad:
            cost = cost/torch.abs(cost)*max_grad
    cost.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    optimizer.zero_grad()
    #if (epoch >= 2/3*n_epochs) and sym == "C4":
    #    print("Use spatial symmetry from now.")
    #    sym = None
    #    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #    n = 0
    Eloc = Elocs.mean().detach()
    Eloc_var = (Elocs).var(axis=0) 
    N = N.mean(axis=0)
    end = timeit.default_timer()
    if epoch%10 == 0 or epoch == 1:
        sx = o.get_sx(samples, log_probs, phases, model, device)
        sy = o.get_sy(samples, log_probs, phases, model, device)
        sz = o.get_sz(samples, model, device)
        print('Epoch: {}/ {}/ t/epoch={}.............'.format(epoch, n_epochs, round(end-start,2)), end=' ')
        print("Loss: {:.8f}".format(cost)+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var)+", Sx: {:.4f}".format(sx)+", Sy: {:.4f}".format(sy)+", Sz: {:.4f}".format(sz)+", N/N_tot: {:.4f} %".format(N.numpy()/N_tot))
    if epoch != n_epochs: del cost, samples, log_probs, phases, Eloc
# ----------- save -----------------------------------
print(samples[:10])
save(model, bounds, fol, device)
np.save(fol+"/Eloc.npy", np.array(Eloc))
np.save(fol+"/N.npy", np.array(N.numpy()))
