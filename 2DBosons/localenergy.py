import torch
import timeit 

def XXZ2D_MatrixElements(Jp1, Jz1, Jp2, Jz2, samples, length_x, length_y, device):
    """ 
    Calculate the local energies of 2D XXZ model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - samples: (num_samples, N)
    - Jp: float
    - Jz: float
    - length_x: system length in x dir
    - length_y: system length in y dir
    """

    Nx         = samples.size()[1]
    Ny         = samples.size()[2]
    numsamples = samples.size()[0]

    #diagonal elements
    diag_matrixelements = torch.zeros((numsamples)).to(device) 
    #diagonal elements from the SzSz term 
    for i in range(Nx): 
        for j in range(Ny):
            if i != length_x:
                values = samples[:,i,j] + samples[:,(i+1)%Nx,j]
                valuesT = values.clone()
                valuesT[values==2]      = 1  #If both spins are up
                valuesT[values==0]      = 1  #If both spins are down
                valuesT[values==1]      = -1 #If they are opposite 
                diag_matrixelements    += valuesT.reshape((numsamples))*Jz1*0.25
            if j != length_y:
                values = samples[:,i,j] + samples[:,i,(j+1)%Ny]
                valuesT = values.clone()
                valuesT[values==2]      = 1  #If both spins are up
                valuesT[values==0]      = 1  #If both spins are down
                valuesT[values==1]      = -1 #If they are opposite
                diag_matrixelements    += valuesT.reshape((numsamples))*Jz1*0.25
            if i != length_x and j!= length_y:
                values = samples[:,i,j] + samples[:,(i+1)%Nx,(j+1)%Ny]
                valuesT = values.clone()
                valuesT[values==2]      = 1  #If both spins are up
                valuesT[values==0]      = 1  #If both spins are down
                valuesT[values==1]      = -1 #If they are opposite
                diag_matrixelements    += valuesT.reshape((numsamples))*Jz2*0.25    

    #off-diagonal elements from the S+S- terms
    if Jp2 != 0:
        offd_matrixelements = torch.zeros((numsamples, (Nx*Ny*3-(Nx+Ny)))).to(device) 
        xprime = torch.zeros((Nx*Ny*3-(Nx+Ny),numsamples, Nx, Ny)).to(device) 
    else:
        offd_matrixelements = torch.zeros((numsamples, (Nx*Ny*2-(Nx+Ny)))).to(device)
        xprime = torch.zeros((Nx*Ny*2-(Nx+Ny),numsamples, Nx, Ny)).to(device)
    if Jp1 != 0 or Jp2 != 0:
        num = 0
        for i in range(Nx): 
            for j in range(Ny):
                if i != length_x:
                    new_samples = samples.clone()
                    values = samples[:,i,j] + samples[:,(i+1)%Nx,j]
                    valuesT = values.clone()
                    new_samples[:,(i+1)%Nx,j]   = samples[:,i,j]
                    new_samples[:,i,j]          = samples[:,(i+1)%Nx, j]
                    valuesT[values==2]      = 0 #If both spins are up
                    valuesT[values==0]      = 0 #If both spins are down
                    valuesT[values==1]      = 1 #If they are opposite   
                    offd_matrixelements[:,num] = valuesT.reshape((numsamples))*Jp1*0.5
                    xprime[num,:]              = new_samples
                    num +=1
                if j != length_y:
                    new_samples = samples.clone()
                    values = samples[:,i,j] + samples[:,i,(j+1)%Ny]
                    valuesT = values.clone()
                    new_samples[:,i,(j+1)%Ny]   = samples[:,i,j]
                    new_samples[:,i,j]          = samples[:,i, (j+1)%Ny]
                    valuesT[values==2]      = 0 #If both spins are up
                    valuesT[values==0]      = 0 #If both spins are down
                    valuesT[values==1]      = 1 #If they are opposite   
                    offd_matrixelements[:,num] = valuesT.reshape((numsamples))*Jp1*0.5
                    xprime[num,:]              = new_samples
                    num +=1
                if Jp2 != 0:
                    if i != length_x and j != length_y:
                        new_samples = samples.clone()
                        values = samples[:,i,j] + samples[:,(i+1)%Nx,(j+1)%Ny]
                        valuesT = values.clone()
                        new_samples[:,(i+1)%Nx,(j+1)%Ny]   = samples[:,i,j]
                        new_samples[:,i,j]                 = samples[:,(i+1)%Nx, (j+1)%Ny]
                        valuesT[values==2]      = 0 #If both spins are up
                        valuesT[values==0]      = 0 #If both spins are down
                        valuesT[values==1]      = 1 #If they are opposite   
                        offd_matrixelements[:,num] = valuesT.reshape((numsamples))*Jp2*0.5
                        xprime[num,:]              = new_samples
                        num +=1

    return diag_matrixelements, offd_matrixelements, xprime


def XXZ2D_Eloc(Jp1, Jz1, Jp2, Jz2, samples, RNN, boundaries, symmetry):
    """ 
    Calculate the local energies of 2D XXZ model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - sample: (num_samples, N)
    - Jp: float
    - Jz: float
    - RNN: RNN model
    - boundaries: str, open or periodic
    """
    device = RNN.device

    Nx         = samples.size()[1]
    Ny         = samples.size()[2]
    numsamples = samples.size()[0]
    if boundaries == "periodic":
        length_x = Nx
        length_y = Ny
    elif "open":
        length_x = Nx-1
        length_y = Ny-1
    else:
        raise "Boundary "+boundaries+" not implemented"
    
    #matrix elements
    diag_me, offd_me, new_samples = XXZ2D_MatrixElements(Jp1, Jz1, Jp2, Jz2, samples, length_x, length_y, device)
    offd_me = offd_me.to(torch.complex64)
    # diagonal elements
    Eloc = diag_me.to(torch.complex64)
    
    length = new_samples.size()[0]
    # pass all samples together through the network
    if symmetry!= None:
        symmetric_samples = get_symmetric_samples(samples, symmetry, device)
        queue_samples = torch.zeros(((length+1+1), numsamples, Nx, Ny), dtype = torch.int32).to(device) 
        queue_samples[length+1] = symmetric_samples
    else:
        queue_samples = torch.zeros(((length+1), numsamples, Nx, Ny), dtype = torch.int32).to(device)  
    
    queue_samples[0] = samples
    queue_samples[1:length+1] = new_samples
    queue_samples_reshaped = torch.reshape(queue_samples, [queue_samples.size()[0]*numsamples, Nx, Ny])
    start = timeit.default_timer()
    log_probs, phases = RNN.log_probabilities(queue_samples_reshaped.to(torch.int64))
    end = timeit.default_timer()
    print("Time for probabilities: "+str(end-start))
    log_probs_reshaped = torch.reshape(log_probs, (queue_samples.size()[0],numsamples)).to(torch.complex64)
    phases_reshaped = torch.reshape(phases, (queue_samples.size()[0],numsamples))
    if Jp1 != 0 or Jp2 != 0: #off-diagonal elements
        for i in range(1,(length+1)):
            tot_log_probs = 0.5*(log_probs_reshaped[i,:]-log_probs_reshaped[0,:]).to(torch.complex64)
            tot_log_probs += 1j*(phases_reshaped[i,:]-phases_reshaped[0,:])
            Eloc += offd_me[:,i-1]*(torch.exp(tot_log_probs))
    if symmetry != None:
        return Eloc, log_probs_reshaped[0], phases_reshaped[0], symmetric_samples, log_probs_reshaped[length+1], phases_reshaped[length+1]
    else:
        return Eloc, log_probs_reshaped[0], phases_reshaped[0]

def get_symmetric_samples(samples, symmetry, device):
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
