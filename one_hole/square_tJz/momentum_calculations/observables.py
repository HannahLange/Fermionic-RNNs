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

def get_particle_no(samples, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    N = torch.zeros((samples.size()[0])).to(device)
    for j in range(Ny):
        for i in range(Nx):
            N[samples[:,i,j]==2] += 1
            N[samples[:,i,j]==1] += 1
            N[samples[:,i,j]==3] += 2
    return N.detach()


def get_n_r(samples):
    N = samples.clone()
    N[samples==2]=1
    return N/N[0].sum()

def shift_samples(samples):
    shifted_x = samples.clone()
    shifted_x[:,2:,:] = samples[:,:-2,:]
    shifted_x[:,:2,:] = samples[:,-2:,:]

    shifted_y = samples.clone()
    shifted_y[:,:,2:] = samples[:,:,:-2]
    shifted_y[:,:,:2] = samples[:,:,-2:]

    return shifted_x, shifted_y

def calculate_momentum(samples, log_probs, log_phases, model):
    ns_x, ns_y = shift_samples(samples)
    ns = torch.cat([ns_x, ns_y])
    nprobs, nphases = model.log_probabilities(ns)

    expect_x = torch.exp(0.5*(nprobs[:samples.size(0)]-log_probs)+1j*(nphases[:samples.size(0)]-log_phases))
    expect_y = torch.exp(0.5*(nprobs[samples.size(0):]-log_probs)+1j*(nphases[samples.size(0):]-log_phases))

    expectation_x = (torch.mean(expect_x, axis=0))
    expectation_y = (torch.mean(expect_y, axis=0))

    #print("T", expectation_x, expectation_y)

    Px = torch.log(expectation_x)*1j/2
    Py = torch.log(expectation_y)*1j/2
    return Px, Py, expect_x, expect_y


def get_szsz(samples, log_probs, boundaries, model, device):
    Nx, Ny, length_x, length_y = get_length(samples, boundaries)
    szsz_x = torch.zeros((samples.size()[0], length_x, length_y+1)).to(device) 
    szsz_y = torch.zeros((samples.size()[0], length_x+1, length_y)).to(device) 
    s = samples.clone().detach() 
    s[samples == 1] = -1
    s[samples == 2] = 1
    for j in range(Ny):
        for i in range(Nx):
            if i != length_x:
                szsz_x[:,i,j] = s[:,i,j]*s[:,(i+1)%Nx,j]
            if j != length_y:
                szsz_y[:,i,j] += s[:,i,j]*s[:,i,(j+1)%Ny]
    return [torch.mean(szsz_x, axis=0).detach().cpu()*1/4, torch.mean(szsz_y, axis=0).detach().cpu()*1/4]

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
                sxsx_x[samples[:,i,j]==0,i,j] = 0
                sxsx_x[samples[:,(i+1)%Nx,j]==0,i,j] = 0
            if j != length_y:
                d = [0,1]
                s1 = flip_neighbor_spins(samples, i, j, d, Nx, Ny)
                log_probs1, phases1 = model.log_probabilities(s1)
                sxsx_y[:,i,j] += torch.real(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases)))
                sxsx_y[samples[:,i,j]==0,i,j] = 0
                sxsx_y[samples[:,i,(j+1)%Ny]==0,i,j] = 0
    return [torch.mean(sxsx_x, axis=0).detach().cpu()*1/4, torch.mean(sxsx_y, axis=0).detach().cpu()*1/4]

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
                s1[:,i,j][s1[:,i,j] == 2] = -1j
                s1[:,i,j][s1[:,i,j] == 1] = 1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 2] = -1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 1] = 1j
                sysy_x[:,i,j] += torch.real(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))*s1[:,i,j]*s1[:,(i+d[0])%Nx,(j+d[1])%Ny])
            if j != length_y:
                d = [0,1]
                s1 = flip_neighbor_spins(samples, i, j, d, Nx, Ny)
                log_probs1, phases1 = model.log_probabilities(s1)
                s1 = s1.to(torch.complex64)
                s1[:,i,j][s1[:,i,j] == 2] = -1j
                s1[:,i,j][s1[:,i,j] == 1] = 1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 2] = -1j
                s1[:,(i+d[0])%Nx,(j+d[1])%Ny][s1[:,(i+d[0])%Nx,(j+d[1])%Ny] == 1] = 1j
                sysy_y[:,i,j] += torch.real(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))*s1[:,i,j]*s1[:,(i+d[0])%Nx,(j+d[1])%Ny])

    return [torch.mean(sysy_x, axis=0).detach().cpu()*1/4, torch.mean(sysy_y, axis=0).detach().cpu()*1/4]

def get_sz_(samples, model, device):
    # used in the cost function, no averaging here!
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sz = torch.zeros((samples.size()[0])).to(device).to(torch.float32) 
    for i in range(Nx):
        for j in range(Ny):
            sz[samples[:,i,j] == 1] += -1/2   
            sz[samples[:,i,j] == 2] += 1/2
    return sz

def get_sz(samples, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sz = torch.zeros((samples.size()[0])).to(device).to(torch.float32) 
    for i in range(Nx):
        for j in range(Ny):
            sz[samples[:,i,j] == 1] += -1/2
            sz[samples[:,i,j] == 2] += 1/2
    return torch.mean(torch.abs(sz)) 

def get_sx(samples, log_probs, phases, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sx = torch.zeros((samples.size()[0], Nx, Ny)).to(torch.complex64).to(device) 
    for i in range(Nx):
        for j in range(Ny):
            s1 = flip_spin(samples, i,j)
            log_probs1, phases1 = model.log_probabilities(s1)
            sx[:,i,j] = (torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases)))
            sx[samples[:,i,j]==0,i,j] = 0
    return (torch.mean(torch.abs(torch.sum(torch.sum(torch.real(sx), axis=-1), axis=-1)*1/2))) 


def get_sx_(samples, log_probs, phases, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sx = torch.zeros((samples.size()[0],)).to(torch.complex64).to(device) 
    for i in range(Nx):
        for j in range(Ny):
            s1 = flip_spin(samples, i,j)
            log_probs1, phases1 = model.log_probabilities(s1)
            sx[samples[:,i,j]!=0] += (0.5*(torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases)))[samples[:,i,j]!=0])
    sx = torch.abs(torch.real(sx))
    return sx



def get_sy(samples, log_probs, phases, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sy = torch.zeros((samples.size()[0], Nx, Ny)).to(torch.complex64).to(device) 
    for i in range(Nx):
        for j in range(Ny):
            s1 = flip_spin(samples, i,j)
            log_probs1, phases1 = model.log_probabilities(s1)
            s1 = s1.to(torch.complex64)
            s1[s1 == 2] = -1j*0.5
            s1[s1 == 1] = 1j*0.5
            sy[:,i,j] = (torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))*s1[:,i,j])
    return (torch.mean(torch.abs(torch.sum(torch.sum(torch.real(sy), axis=-1), axis=-1))))


def get_sy_(samples, log_probs, phases, model, device):
    Nx = samples.size()[1]
    Ny = samples.size()[2]
    sy = torch.zeros((samples.size()[0],)).to(device).to(torch.complex64)
    for i in range(Nx):
        for j in range(Ny):
            s1 = flip_spin(samples, i,j)
            log_probs1, phases1 = model.log_probabilities(s1)
            s1 = s1.to(torch.complex64)[:,i,j]
            s1[s1 == 2] = -1j*0.5
            s1[s1 == 1] = 1j*0.5
            sy += torch.exp(0.5*(log_probs1-log_probs))*torch.exp(1j*(phases1-phases))*s1
    sy = torch.abs(torch.real(sy))
    return sy


def flip_neighbor_spins(samples, i,j, direction, Nx, Ny):
    s = samples.clone().detach()
    N = s.size()[1]
    s[:,i,j][samples[:,i,j] == 1]   = 2
    s[:,i,j][samples[:,i,j] == 2]   = 1
    s[:,(i+direction[0])%Nx,(j+direction[1])%Ny][samples[:,(i+direction[0])%Nx,(j+direction[1])%Ny] == 1] = 2
    s[:,(i+direction[0])%Nx,(j+direction[1])%Ny][samples[:,(i+direction[0])%Nx,(j+direction[1])%Ny] == 2] = 1
    return s

def flip_spin(samples, i,j):
    s = samples.clone().detach()
    s[:,i,j][samples[:,i,j] == 1] = 2
    s[:,i,j][samples[:,i,j] == 2] = 1
    return s


