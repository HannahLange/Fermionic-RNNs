import torch
import numpy as np

def tJ2D_MatrixElements(Jp, Jz, t, samples_, length_x, length_y, antisym, device, dtype=torch.float32):
    """ 
    Calculate the local energies of 2D t-J model given a set of set of samples.
    Returns: The local energies (diagonal and offdiagonal), the connected samples and the respective sign.
    Inputs:
    - samples: (num_samples, N)
    - t: float
    - Jp: float
    - Jz: float
    - length_x: system length in x dir
    - length_y: system length in y dir
    - antisym: bosons if False and fermions if True
    - device and dtype
    """

    Nx         = samples_.size()[1]
    Ny         = samples_.size()[2]
    numsamples = samples_.size()[0]

    if length_x == Nx and length_y == Ny:
        l = Nx*Ny*2
    elif length_x == Nx:
        l = Nx*Ny*2-Nx
    elif length_y == Ny:
        l = Nx*Ny*2-Ny
    else:
        l = (Nx*Ny*2-(Nx+Ny))

    # ---------- hopping term ----------------
    matrixelements = torch.zeros((numsamples, l), dtype=dtype, device=device)
    xprime_t = torch.zeros((l,numsamples, Nx, Ny), dtype=dtype, device=device)
    signs = torch.zeros((l,numsamples), dtype=dtype, device=device)
    num = 0
    samples = samples_.detach().clone()
    samples[samples_==1] = -1 #rename spin down for convenienve
    if t!= 0:
        for i in range(Nx):
            for j in range(Ny):  
                if i != length_x:
                    values = samples[:,i,j] + samples[:,(i+1)%Nx,j]
                    new_samples = samples.clone()
                    new_samples[:,(i+1)%Nx,j] = samples[:,i,j]  
                    new_samples[:,i,j]        = samples[:,(i+1)%Nx, j]
                    valuesT = values.clone()
                    #If both are occupied
                    valuesT[values==4]   = 0  #If both spins are up
                    valuesT[values==-2]  = 0  #If both spins are down
                    valuesT[values==1]   = 0  #If they are opposite
                    #If one is occupied
                    valuesT[values==2]   = -1
                    valuesT[values==-1]  = -1
                    matrixelements[:,num]  += valuesT.reshape((numsamples))*t
                    xprime_t[num,:] = new_samples
                    num += 1
                if j != length_y:
                    values = samples[:,i,j] + samples[:,i,(j+1)%Ny]
                    new_samples = samples.clone()
                    new_samples[:,i,(j+1)%Ny]   = samples[:,i,j]
                    new_samples[:,i,j]          = samples[:,i, (j+1)%Ny]
                    valuesT = values.clone()
                    #If both are occupied
                    valuesT[values==4]   = 0  #If both spins are up
                    valuesT[values==-2]  = 0  #If both spins are down
                    valuesT[values==1]   = 0  #If they are opposite  
                    #If one is occupied
                    valuesT[values==2]   = -1
                    valuesT[values==-1]  = -1
                    matrixelements[:,num] +=valuesT.reshape((numsamples))*t
                    xprime_t[num,:] = new_samples
                    # Here I have to take care of the sign changes!
                    # 1. hopping in x direction means that I will have to 
                    # exchange particles to restore the order
                    # Note: needs to be modified for more than one hole and periodic boundaries!!!
                    P = torch.zeros((numsamples), dtype=dtype, device=device)
                    if antisym:
                        if j % 2 == 0: #go from left to right
                            for i2 in range(i+1,Nx):
                                p1 = samples[:,i2,j].clone()
                                p2 = samples[:,i2,(j+1)%Ny].clone()
                                p1[samples[:,i2,j]==-1] = 0
                                p1[samples[:,i2,j]==2] = 0
                                p1[samples[:,i2,j]==0] = 1
                                p2[samples[:,i2,(j+1)%Ny]==-1] = 0
                                p2[samples[:,i2,(j+1)%Ny]==2] = 0
                                p2[samples[:,i2,(j+1)%Ny]==0] = 1
                                P += (p1+p2)
                        elif j % 2 == 1: #go from right to left    
                            for i2 in range(i-1,-1,-1):
                                p1 = samples[:,i2,j].clone()
                                p2 = samples[:,i2,(j+1)%Ny].clone()
                                p1[samples[:,i2,j]==-1] = 0
                                p1[samples[:,i2,j]==2] = 0
                                p1[samples[:,i2,j]==0] = 1
                                p2[samples[:,i2,(j+1)%Ny]==-1] = 0
                                p2[samples[:,i2,(j+1)%Ny]==2] = 0
                                p2[samples[:,i2,(j+1)%Ny]==0] = 1
                                P += (p1+p2)
                        P[valuesT==0] = 0
                        signs[num] = P*np.pi

                    num += 1

    # ---------- S_i* S_j term ---------------

    #diagonal elements from the SzSz term 
    diag_matrixelements = torch.zeros((numsamples), dtype=dtype, device=device) 
    for i in range(Nx): 
        for j in range(Ny):
            if i != length_x:
                values = samples[:,i,j] + samples[:,(i+1)%Nx,j]
                valuesT = values.clone()
                valuesT[values==4]   = 1  #If both spins are up
                valuesT[values==-2]  = 1  #If both spins are down
                valuesT[values==1]   = -1 #If they are opposite 
                #If one is occupied
                valuesT[values==2]   = 0
                valuesT[values==-1]  = 0
                diag_matrixelements    += valuesT.reshape((numsamples))*Jz*0.25
            if j != length_y:
                values = samples[:,i,j] + samples[:,i,(j+1)%Ny]
                valuesT = values.clone()
                valuesT[values==4]   = 1  #If both spins are up
                valuesT[values==-2]  = 1  #If both spins are down
                valuesT[values==1]   = -1 #If they are opposite  
                #If one is occupied
                valuesT[values==2]   = 0
                valuesT[values==-1]  = 0
                diag_matrixelements    += valuesT.reshape((numsamples))*Jz*0.25

    
    #off-diagonal elements from the S+S- terms
    offd_matrixelements = torch.zeros((numsamples, l), dtype=dtype, device=device)
    xprime = torch.zeros((l, numsamples, Nx, Ny), dtype=dtype, device=device)
    num = 0
    
    for i in range(Nx): 
        for j in range(Ny):
            if i != length_x:
                new_samples = samples.clone()
                values = samples[:,i,j] + samples[:,(i+1)%Nx,j]
                valuesT = values.clone()
                new_samples[:,(i+1)%Nx,j]   = samples[:,i,j]
                new_samples[:,i,j]          = samples[:,(i+1)%Nx, j]
                valuesT[values==4]   = 0 #If both spins are up
                valuesT[values==-2]  = 0 #If both spins are down
                valuesT[values==1]   = 1 #If they are opposite 
                #If one is occupied
                valuesT[values==2]   = 0
                valuesT[values==-1]  = 0
                offd_matrixelements[:,num] = valuesT.reshape((numsamples))*Jp*0.5
                xprime[num,:]              = new_samples
                num +=1
            if j != length_y:
                new_samples = samples.clone()
                values = samples[:,i,j] + samples[:,i,(j+1)%Ny]
                valuesT = values.clone()
                new_samples[:,i,(j+1)%Ny]   = samples[:,i,j]
                new_samples[:,i,j]          = samples[:,i, (j+1)%Ny]
                valuesT[values==4]   = 0 #If both spins are up
                valuesT[values==-2]  = 0 #If both spins are down
                valuesT[values==1]   = 1 #If they are opposite
                #If one is occupied
                valuesT[values==2]   = 0
                valuesT[values==-1]  = 0
                offd_matrixelements[:,num] = valuesT.reshape((numsamples))*Jp*0.5
                xprime[num,:]              = new_samples
                num +=1

    # ---------- n_i*n_j  ----------------
    for i in range(Nx):
        for j in range(Ny):
            if i != length_x:
                values = samples[:,i,j] + samples[:,(i+1)%Nx,j]
                valuesT = values.clone()
                valuesT[values==4]   = 1  #If both spins are up
                valuesT[values==-2]  = 1  #If both spins are down
                valuesT[values==1]   = 1 #If they are opposite
                #If one is occupied
                valuesT[values==2]   = 0
                valuesT[values==-1]  = 0
                diag_matrixelements  -= valuesT.reshape((numsamples))*Jz/4
            if j != length_y:
                values = samples[:,i,j] + samples[:,i,(j+1)%Ny]
                valuesT = values.clone()
                valuesT[values==4]   = 1  #If both spins are up
                valuesT[values==-2]  = 1  #If both spins are down
                valuesT[values==1]   = 1 #If they are opposite
                #If one is occupied
                valuesT[values==2]   = 0
                valuesT[values==-1]  = 0
                diag_matrixelements  -= valuesT.reshape((numsamples))*Jz/4
    if t != 0:
        offd_matrixelements = torch.concat([offd_matrixelements, matrixelements], axis=1)
        xprime              = torch.concat([xprime, xprime_t], axis=0)
    x_clone = xprime.clone()
    xprime[x_clone==-1] = 1 #reverse the relabeling of spin downs in the beginning of the function
    return diag_matrixelements, offd_matrixelements, xprime, signs


def get_Eloc(model, parameters, samples, RNN, boundaries_x, boundaries_y, symmetry, antisym):
    """ 
    Calculate the local energies of 2D t-J model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - model: str ("tJ")
    - parameters: dict of parameters of the model
    - samples: (num_samples, N)
    - RNN: RNN model
    - boundaries: str, open or periodic
    """

    device = RNN.device

    Nx         = samples.size()[1]
    Ny         = samples.size()[2]
    numsamples = samples.size()[0]
    if boundaries_x == "periodic":
        length_x = Nx
    elif boundaries_x == "open":
        length_x = Nx-1
    if boundaries_y == "periodic":
        length_y = Ny
    elif boundaries_y == "open":
        length_y = Ny-1
    else:
        raise "Boundary "+boundaries_x+"/"+boundaries_y+" not implemented"

    #matrix elements
    if model == "tJ":
        Jp = parameters["Jp"]
        Jz = parameters["Jz"]
        t  = parameters["t"]
        diag_me, offd_me, new_samples, signs = tJ2D_MatrixElements(Jp, Jz, t, samples, length_x, length_y, antisym, device)
    else:
        raise NotImplementedError

    offd_me = offd_me.to(torch.complex64)
    # diagonal elements
    Eloc = diag_me.to(torch.complex128)

    length = new_samples.size()[0]
    # pass all samples together through the network
    if symmetry!= None: #get the symmetric samples
        symmetric_samples = get_symmetric_samples(samples, symmetry, device)
        queue_samples = torch.zeros(((length+1+1), numsamples, Nx, Ny), dtype=torch.int32, device=device) 
        queue_samples[length+1] = symmetric_samples
    else:
        queue_samples = torch.zeros(((length+1), numsamples, Nx, Ny), dtype=torch.int32, device=device)  
    queue_samples[0] = samples
    queue_samples[1:length+1] = new_samples
    queue_samples_reshaped = torch.reshape(queue_samples, [queue_samples.size()[0]*numsamples, Nx, Ny])
    log_probs, phases = RNN.log_probabilities(queue_samples_reshaped.to(torch.int64))
    log_probs_reshaped = torch.reshape(log_probs, (queue_samples.size()[0],numsamples))
    phases_reshaped = torch.reshape(phases, (queue_samples.size()[0],numsamples))
    if t!= 0:
        phases_reshaped[int(length/2+1):length+1,:] = signs + phases_reshaped[int(length/2+1):length+1,:]
    for i in range(1,(length+1)):
        log_ampl = 0.5*(log_probs_reshaped[i,:]-log_probs_reshaped[0,:])
        phase    = (phases_reshaped[i,:]-phases_reshaped[0,:])
        Eloc += offd_me[:,i-1]*(torch.exp(torch.complex(log_ampl, phase)))
    if symmetry != None:
        return Eloc, log_probs_reshaped[0], phases_reshaped[0], symmetric_samples, log_probs_reshaped[length+1], phases_reshaped[length+1]
    else:
        return Eloc, log_probs_reshaped[0], phases_reshaped[0]




def get_symmetric_samples(original_samples, symmetry, device):
    """
        Rotates the samples according to the specified symmetry.
    """
    samples = original_samples.clone()
    if symmetry == "C4":
        symmetric_samples = torch.zeros((samples.size()[0],samples.size()[1], samples.size()[2]), dtype = torch.int32).to(device)  
        l = int(samples.size()[0]/3)
        symmetric_samples[:l]    = torch.rot90(samples[:l],    1, [1,2])
        symmetric_samples[l:2*l] = torch.rot90(samples[l:2*l], 2, [1,2])
        symmetric_samples[2*l:]  = torch.rot90(samples[2*l:],  3, [1,2])
    if symmetry == "C2":
        symmetric_samples = torch.zeros((samples.size()[0],samples.size()[1], samples.size()[2]), dtype = torch.int32).to(device)
        symmetric_samples = torch.rot90(samples, 2, [1,2])
    return symmetric_samples

