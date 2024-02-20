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



