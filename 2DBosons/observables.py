import torch
import numpy as np


# observables that can be evaluated during the training or afterwards
def get_length(samples, boundaries):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    if boundaries == "periodic":
        length_x = Nx
        length_y = Ny
    else:
        length_x = Nx-1
        length_y = Ny-1
    return Nx, Ny, length_x, length_y


def get_szsz(samples, log_probs, boundaries, model, device):
    Nx, Ny, length_x, length_y = get_length(samples, boundaries)
    szsz_x = torch.zeros((samples.size()[0], length_x, length_y+1)).to(device) 
    szsz_y = torch.zeros((samples.size()[0], length_x+1, length_y)).to(device) 
    s = samples.clone().detach() 
    s[samples == 0] = -1
    for j in range(Ny):
        for i in range(Nx):
            if i != length_x:
                szsz_x[:,i,j] = s[:,i,j]*s[:,(i+1)%Nx,j]
            if j != length_y:
                szsz_y[:,i,j] += s[:,i,j]*s[:,i,(j+1)%Ny]
    return [torch.mean(szsz_x, axis=0).detach().numpy()*1/4, torch.mean(szsz_y, axis=0).detach().numpy()*1/4]

def get_sxsx(samples, log_probs, phases, boundaries, model, device):
    Nx, Ny, length_x, length_y = get_length(samples, boundaries)
    sxsx_x = torch.zeros((samples.size()[0], length_x, length_y+1)).to(device) 
    sxsx_y = torch.zeros((samples.size()[0], length_x+1, length_y)).to(device) 
    for j in range(Ny):
        for i in range(Nx):
            if i != length_x:
                d = [1,0]
                s1 = flip_neighbor_spins(samples, i, j, d, Nx, Ny)
                log_probs1, phases1 = model.log_probabilities(s1)
                sxsx_x[:,i,j] += torch.real(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases)))
            if j != length_y:
                d = [0,1]
                s1 = flip_neighbor_spins(samples, i, j, d, Nx, Ny)
                log_probs1, phases1 = model.log_probabilities(s1)
                sxsx_y[:,i,j] += torch.real(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases)))
    return [torch.mean(sxsx_x, axis=0).detach().numpy()*1/4, torch.mean(sxsx_y, axis=0).detach().numpy()*1/4]

def get_sysy(samples, log_probs, phases, boundaries, model, device):
    Nx, Ny, length_x, length_y = get_length(samples, boundaries)
    sysy_x = torch.zeros((samples.size()[0], length_x, length_y+1)).to(device) 
    sysy_y = torch.zeros((samples.size()[0], length_x+1, length_y)).to(device) 
    for j in range(Ny):
        for i in range(Nx):
            if i != length_x:
                d = [1,0]
                s1 = flip_neighbor_spins(samples, i, j, d, Nx, Ny)
                log_probs1, phases1 = model.log_probabilities(s1)
                s1 = s1.to(torch.complex64)
                s1[:,i,j][s1[:,i,j] == 1] = -1j
                s1[:,i,j][s1[:,i,j] == 0] = 1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 1] = -1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 0] = 1j
                sysy_x[:,i,j] += torch.real(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))*s1[:,i,j]*s1[:,(i+d[0])%Nx,(j+d[1])%Ny])
            if j != length_y:
                d = [0,1]
                s1 = flip_neighbor_spins(samples, i, j, d, Nx, Ny)
                log_probs1, phases1 = model.log_probabilities(s1)
                s1 = s1.to(torch.complex64)
                s1[:,i,j][s1[:,i,j] == 1] = -1j
                s1[:,i,j][s1[:,i,j] == 0] = 1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 1] = -1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 0] = 1j
                sysy_y[:,i,j] += torch.real(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))*s1[:,i,j]*s1[:,(i+d[0])%Nx,(j+d[1])%Ny])

    return [torch.mean(sysy_x, axis=0).detach().numpy()*1/4, torch.mean(sysy_y, axis=0).detach().numpy()*1/4]

def get_sz_(samples, model, device):
    # used in the cost function, no averaging here!
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sz = torch.zeros((samples.size()[0], Nx, Ny)).to(device) 
    s = samples.clone().detach() 
    s[samples == 0] = -1
    sz = s.to(torch.float64)
    return torch.sum(torch.sum(sz, axis=2), axis=1) *1/2 

def get_sz(samples, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sz = torch.zeros((samples.size()[0], Nx, Ny)).to(device) 
    s = samples.clone().detach() 
    s[samples == 0] = -1
    sz = s.to(torch.float64)
    return torch.sum(torch.mean(sz, axis=0)*1/2) / (Nx*Ny)

def get_sx(samples, log_probs, phases, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sx = torch.zeros((samples.size()[0], Nx, Ny)).to(device) 
    for i in range(Nx):
        for j in range(Ny):
            s1 = flip_spin(samples, i,j)
            log_probs1, phases1 = model.log_probabilities(s1)
            sx[:,i,j] = torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))
    return torch.sum(torch.mean(sx, axis=0)*1/2) / (Nx*Ny)

def get_sy(samples, log_probs, phases, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sy = torch.zeros((samples.size()[0], Nx, Ny)).to(device) 
    for i in range(Nx):
        for j in range(Ny):
            s1 = flip_spin(samples, i,j)
            log_probs1, phases1 = model.log_probabilities(s1)
            s1 = s1.to(torch.complex64)
            s1[:,i,j][s1[:,i,j] == 1] = -1j
            s1[:,i,j][s1[:,i,j] == 0] = 1j
            sy[:,i,j] = torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))*s1[:,i,j]
    return torch.sum(torch.mean(sy, axis=0)*1/2) / (Nx*Ny)


def flip_neighbor_spins(samples, i,j, direction, Nx, Ny):
    s = samples.clone().detach()
    N = s.size()[1]
    s[:,i,j][samples[:,i,j] == 0]   = 1
    s[:,i,j][samples[:,i,j] == 1]   = 0
    s[:,(i+direction[0])%Nx,(j+direction[1])%Ny][samples[:,(i+direction[0])%Nx,(j+direction[1])%Ny] == 0] = 1
    s[:,(i+direction[0])%Nx,(j+direction[1])%Ny][samples[:,(i+direction[0])%Nx,(j+direction[1])%Ny] == 1] = 0
    return s

def flip_spin(samples, i,j):
    s = samples.clone().detach()
    s[:,i,j][samples[:,i,j] == 0] = 1
    s[:,i,j][samples[:,i,j] == 1] = 0
    return s
