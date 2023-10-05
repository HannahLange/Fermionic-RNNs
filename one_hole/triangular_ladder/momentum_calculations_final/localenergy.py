import torch
import numpy as np

def tJ2D_MatrixElements(Jp, Jz, t, samples_, length_x, length_y, antisym, device, dtype=torch.float32):
    """ 
    Calculate the local energies of 2D t-J model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - samples: (num_samples, N)
    - t: float
    - Jp: float
    - Jz: float
    - length_x: system length in x dir
    - length_y: system length in y dir
    """

    Nx         = samples_.size()[1]
    Ny         = samples_.size()[2]
    numsamples = samples_.size()[0]
    l = Nx*Ny+Ny*(Nx-1)*2
    # ---------- hopping term ----------------
    if antisym:
        raise NotImplementedError
    matrixelements = torch.zeros((numsamples, l), dtype=dtype, device=device)
    xprime_t = torch.zeros((l,numsamples, Nx, Ny), dtype=dtype, device=device)
    signs = torch.zeros((l,numsamples), dtype=dtype, device=device)
    num = 0
    samples = samples_.detach().clone()
    samples[samples_==1] = -1
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
                    valuesT[values==1]   = 0 #If they are opposite
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
                    valuesT[values==1]   = 0 #If they are opposite  
                    #If one is occupied
                    valuesT[values==2]   = -1
                    valuesT[values==-1]  = -1
                    matrixelements[:,num] +=valuesT.reshape((numsamples))*t
                    xprime_t[num,:] = new_samples
                    num += 1
                if i != length_x and j != length_y:
                    values = samples[:,i,j] + samples[:,(i+1)%Nx,(j+1)%Ny]
                    new_samples = samples.clone()
                    new_samples[:,(i+1)%Nx,(j+1)%Ny]   = samples[:,i,j]
                    new_samples[:,i,j]          = samples[:,(i+1)%Nx,(j+1)%Ny]
                    valuesT = values.clone()
                    #If both are occupied
                    valuesT[values==4]   = 0  #If both spins are up
                    valuesT[values==-2]  = 0  #If both spins are down
                    valuesT[values==1]   = 0 #If they are opposite  
                    #If one is occupied
                    valuesT[values==2]   = -1
                    valuesT[values==-1]  = -1
                    matrixelements[:,num] +=valuesT.reshape((numsamples))*t
                    xprime_t[num,:] = new_samples
                    num += 1

    # ---------- S_i* S_j term ---------------
    #diagonal elements
    diag_matrixelements = torch.zeros((numsamples), dtype=dtype, device=device) 
    #diagonal elements from the SzSz term 
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
            if i != length_x and j!= length_y:
                values = samples[:,i,j] + samples[:,(i+1)%Nx,(j+1)%Ny]
                valuesT = values.clone()
                valuesT[values==4]      = 1  #If both spins are up
                valuesT[values==-2]      = 1  #If both spins are down
                valuesT[values==1]      = -1 #If they are opposite
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
            if i != length_x and j != length_y:
                new_samples = samples.clone()
                values = samples[:,i,j] + samples[:,(i+1)%Nx,(j+1)%Ny]
                valuesT = values.clone()
                new_samples[:,(i+1)%Nx,(j+1)%Ny]   = samples[:,i,j]
                new_samples[:,i,j]                 = samples[:,(i+1)%Nx, (j+1)%Ny]
                valuesT[values==4]      = 0 #If both spins are up
                valuesT[values==-2]      = 0 #If both spins are down
                valuesT[values==1]      = 1 #If they are opposite   
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
            if i != length_x and j != length_y:
                values = samples[:,i,j] + samples[:,(i+1)%Nx,(j+1)%Ny]
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
    xprime[x_clone==-1] = 1
    return diag_matrixelements, offd_matrixelements, xprime, signs


def get_Eloc(model, parameters, samples, RNN, boundaries_x, boundaries_y, symmetry, antisym):
    """ 
    Calculate the local energies of 2D t-J model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - model: str (tJ or FH)
    - parameters: dict of parameters of the model
    - samples: (num_samples, N)
    - RNN: RNN model
    - boundaries: str, open or periodic
    """
    if model == "tJ":
        Jp = parameters["Jp"]
        Jz = parameters["Jz"]
        t  = parameters["t"]
    elif model == "FH":
        U = parameters["U"]
        t  = parameters["t"]
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
        diag_me, offd_me, new_samples, signs = tJ2D_MatrixElements(Jp, Jz, t, samples, length_x, length_y, antisym, device)
    elif model == "FH":
        diag_me, offd_me, new_samples, signs = FH2D_MatrixElements(U, t, samples, length_x, length_y, antisym, device)

    offd_me = offd_me.to(torch.complex64)
    # diagonal elements
    Eloc = diag_me.to(torch.complex128)

    length = new_samples.size()[0]
    # pass all samples together through the network
    if symmetry!= None:
        symmetric_samples = get_symmetric_samples(samples, symmetry, device)
        queue_samples = torch.zeros(((length+1+1), numsamples, Nx, Ny), dtype=torch.int32, device=device) 
        queue_samples[length+1] = symmetric_samples
    else:
        queue_samples = torch.zeros(((length+1), numsamples, Nx, Ny), dtype=torch.int32, device=device)  
    #Eloc2 = diag_me.to(torch.complex64)
    queue_samples[0] = samples
    queue_samples[1:length+1] = new_samples
    queue_samples_reshaped = torch.reshape(queue_samples, [queue_samples.size()[0]*numsamples, Nx, Ny])
    log_probs, phases = RNN.log_probabilities(queue_samples_reshaped.to(torch.int64))
    log_probs_reshaped = torch.reshape(log_probs, (queue_samples.size()[0],numsamples))
    phases_reshaped = torch.reshape(phases, (queue_samples.size()[0],numsamples))
    # add the signs due to fermonic exchange
    if t!= 0 and model == "tJ":
        phases_reshaped[int(length/2+1):length+1,:] = signs + phases_reshaped[int(length/2+1):length+1,:]
    for i in range(1,(length+1)):
        log_ampl = 0.5*(log_probs_reshaped[i,:]-log_probs_reshaped[0,:])
        phase    = (phases_reshaped[i,:]-phases_reshaped[0,:])
        Eloc += offd_me[:,i-1]*(torch.exp(torch.complex(log_ampl, phase)))
        #Eloc2 += offd_me[:,i-1]
    #print(samples[0])
    if symmetry != None:
        return Eloc, log_probs_reshaped[0], phases_reshaped[0], symmetric_samples, log_probs_reshaped[length+1], phases_reshaped[length+1]
    else:
        return Eloc, log_probs_reshaped[0], phases_reshaped[0]



def FH2D_MatrixElements(U, t, samples, length_x, length_y, antisym, device):
    """ 
    Calculate the local energies of 2D Fermi Hubbard model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - samples: (num_samples, N)
    - t: float
    - U: float
    - length_x: system length in x dir
    - length_y: system length in y dir
    """

    Nx         = samples.size()[1]
    Ny         = samples.size()[2]
    numsamples = samples.size()[0]
    if length_x == Nx or length_y == Ny:
        l = Nx*Ny*2
    else:
        l = (Nx*Ny*2-(Nx+Ny))

    # ---------- hopping term ----------------

    offd_matrixelements = torch.zeros((numsamples, l)).to(device)
    xprime_t = torch.zeros((l,numsamples, Nx, Ny)).to(device)
    signs = torch.zeros((l,numsamples)).to(device)
    num = 0
    samples = samples.detach().clone()
    samples[samples==1] = -1
    samples[samples==3] = 6
    if t!= 0:
        for i in range(Nx):
            for j in range(Ny):  
                if i != length_x:
                    values = samples[:,i,j] + 0.3*samples[:,(i+1)%Nx,j]
                    new_samples = samples.clone()
                    new_samples[:,(i+1)%Nx,j]   = samples[:,i,j]  
                    new_samples[:,i,j]          = samples[:,(i+1)%Nx, j]
                    aux_ten1 = -torch.ones((new_samples[values==6,i,j].size()[0]), device=device).long()
                    aux_ten2 = torch.ones((new_samples[values==6,i,j].size()[0]), device=device).long()*2
                    new_samples[values==6,i,j] = aux_ten1
                    new_samples[values==6,(i+1)%Nx,j] = aux_ten2
                    valuesT = values.clone()
                    valuesT = hopping_amplitudes_FH(valuesT, values)
                    offd_matrixelements[:,num]  += valuesT.reshape((numsamples))*t
                    xprime_t[num,:] = new_samples
                    num += 1
                if j != length_y:
                    values = samples[:,i,j] + 0.3*samples[:,i,(j+1)%Ny]
                    new_samples = samples.clone()
                    new_samples[:,i,(j+1)%Ny]   = samples[:,i,j]  
                    new_samples[:,i,j]          = samples[:,i,(j+1)%Ny]
                    aux_ten1 = -torch.ones((new_samples[values==6,i,j].size()[0]), device=device).long()
                    aux_ten2 = torch.ones((new_samples[values==6,i,j].size()[0]), device=device).long()*2
                    new_samples[values==6,i,j] = aux_ten1
                    new_samples[values==6,i,(j+1)%Ny] = aux_ten2
                    valuesT = values.clone()
                    valuesT = hopping_amplitudes_FH(valuesT, values)
                    offd_matrixelements[:,num]  += valuesT.reshape((numsamples))*t
                    xprime_t[num,:] = new_samples
                    P = torch.zeros((numsamples)).to(device)
                    if antisym:
                        if j % 2 == 0: #go from left to right    
                            for i2 in range(j+1,Nx):
                                p1 = samples[:,i2,j].clone()
                                p2 = samples[:,i2,(j+1)%Ny].clone()
                                p1[samples[:,i2,j]==-1] = 0
                                p1[samples[:,i2,j]==2] = 0
                                p1[samples[:,i2,j]==6] = 1
                                p2[samples[:,i2,(j+1)%Ny]==-1] = 0
                                p2[samples[:,i2,(j+1)%Ny]==2] = 0
                                p2[samples[:,i2,(j+1)%Ny]==6] = 1
                                P += (p1+p2)
                        elif j % 2 == 1: #go from right to left    
                            for i2 in range(j-1,-1,-1):
                                p1 = samples[:,i2,j].clone()
                                p2 = samples[:,i2,(j+1)%Ny].clone()
                                p1[samples[:,i2,j]==-1] = 0
                                p1[samples[:,i2,j]==2] = 0
                                p1[samples[:,i2,j]==6] = 1
                                p2[samples[:,i2,(j+1)%Ny]==-1] = 0
                                p2[samples[:,i2,(j+1)%Ny]==2] = 0
                                p2[samples[:,i2,(j+1)%Ny]==6] = 1
                                P += (p1+p2)
                        P[valuesT==0] = 0
                    num += 1
    # ---------- U term  ----------------
    diag_matrixelements = torch.zeros((numsamples)).to(device) 
    for i in range(Nx):
        for j in range(Ny):
            values = samples[:,i,j]
            valuesT = values.clone()
            valuesT[values==2]   = 0  #If one spin up
            valuesT[values==-1]  = 0  #If one spin down
            valuesT[values==6]   = 1  #If two sit on the same site
            diag_matrixelements  += valuesT.reshape((numsamples))*U
    
    x_clone = x_prime.clone()
    xprime_t[x_clone==-1] = 1
    xprime_t[x_clone==6] = 3
    return diag_matrixelements, offd_matrixelements, xprime_t, signs




def get_symmetric_samples(original_samples, symmetry, device):
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


def hopping_amplitudes_FH(valuesT, values):
    valuesT[values==2.6]  = 0  # up up
    valuesT[values==-1.3] = 0  # dn dn
    valuesT[values==1.7]  = -1  # up dn
    valuesT[values==-0.4] = -1  # dn up
    valuesT[values==3.8]  = 0  # up updn
    valuesT[values==0.8]  = 0  # dn updn
    valuesT[values==6.6]  = -1  # updn up
    valuesT[values==5.7]  = -1  # updn dn
    valuesT[values==6]    = -1  # updn _
    valuesT[values==1.8]  = 0  # _ updn
    valuesT[values==0.6]  = 0  # _ up
    valuesT[values==2]    = -1  # up _
    valuesT[values==-0.3] = 0  # _ dn
    valuesT[values==-1]   = -1  # dn _
    valuesT[values==7.6]  = 0  # updn updn
    valuesT[values==0]    = 0  # _ _
    return valuesT
