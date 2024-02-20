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

def shift_samples(samples, n=1):
    shifted_x = samples.clone()
    shifted_x[:,n:,:] = samples[:,:-n,:]
    shifted_x[:,:n,:] = samples[:,-n:,:]

    shifted_y = samples.clone()
    shifted_y[:,:,n:] = samples[:,:,:-n]
    shifted_y[:,:,:n] = samples[:,:,-n:]

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

    Px = torch.log(expectation_x)*1j
    Py = torch.log(expectation_y)*1j
    return Px, Py, expect_x, expect_y



