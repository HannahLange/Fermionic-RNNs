import torch
from torch import nn
import numpy as np
import timeit
from torch import index_select


torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(1234)

class TensorizedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, dtype=torch.float64, celltype="GRU"):
        super().__init__()
        """
        Custom RNN / GRU layer for 2D input 
            input_size: 2 (spin system), 3 (t-J model), 4 (FH model)
            hidden_size: hidden dimension of the RNN cells
            device: CPU or GPU
            dtype: default: torch.float64
            celltype: plain vanilla "RNN" or gated unit "GRU"
        """

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.sigmoid  = torch.nn.Sigmoid()
        self.tanh     = torch.nn.Tanh()
        self.elu      = torch.nn.ELU()
        self.device   = device
        self.celltype = celltype        
        self.layer    = 1

        # define all layers / weights
        input_dim = 2*self.input_size
        factory_kwargs = {'device': device, 'dtype': dtype}
        if self.celltype == "GRU" or self.celltype == "RNN":
            w1      = torch.empty((input_dim, 2*self.hidden_size, self.hidden_size),**factory_kwargs)
            self.W1 = nn.Parameter(w1)  # nn.Parameter is a Tensor that's a module parameter.
            b1      = torch.empty((self.hidden_size), **factory_kwargs)
            self.b1 = nn.Parameter(b1).to(self.device)
        if self.celltype == "GRU":
            w2      = torch.empty((input_dim, 2*self.hidden_size, self.hidden_size),**factory_kwargs)
            self.W2 = nn.Parameter(w2).to(self.device) 
            b2      = torch.empty((self.hidden_size), **factory_kwargs)
            self.b2 = nn.Parameter(b2).to(self.device)

            wmerge      = torch.empty((2*self.hidden_size, self.hidden_size),**factory_kwargs)
            self.Wmerge = nn.Parameter(wmerge).to(self.device) 

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1, 1)
        if self.celltype == "GRU":
            nn.init.xavier_uniform_(self.W2, 1)
            nn.init.xavier_uniform_(self.Wmerge, 1)
            self.b2.data.fill_(0.01)
        self.b1.data.fill_(0.01)


    def forward(self, inputs, states):
        """
        Passes the inputs through the RNN/ GRU cell.
            inputs: list of 2 input states from previous cells in x and y direction.
            states: hidden states, list of 2 input states from previous cells in x and y direction.
        """
        if len(inputs[0].size()) == 3:
            inputs[0] = inputs[0][:,0,:]
        if len(inputs[1].size()) == 3:
            inputs[1] = inputs[1][:,0,:]

        # calculate sigma * T * h term
        if self.celltype == "RNN":
            inputstate1_ = torch.einsum('ijk,lj->ikl', self.W1, torch.concat((states[0], states[1]), 1))
            if self.layer == 1: inputstate1 = torch.einsum('li,ikl->lk',torch.concat((inputs[0], inputs[1]), 1), inputstate1_)
            else: inputstate1 = torch.einsum('li,ikl->lk',inputs, inputstate1_)
            new_state = self.elu(inputstate1 + self.b1)
        if self.celltype == "GRU":
            inputstate1_ = torch.einsum('ijk,lj->ikl', self.W1, torch.concat((states[0], states[1]), 1))
            if self.layer == 1: inputstate1 = torch.einsum('li,ikl->lk',torch.concat((inputs[0], inputs[1]), 1), inputstate1_)
            else: inputstate1 = torch.einsum('li,ikl->lk',inputs, inputstate1_)
            # calculate h_tilde in Eq. A1 of Hibat Allah 2022
            state_tilda = self.tanh(inputstate1 + self.b1)
            # calculate u_n in Eq. A1
            inputstate2_ = torch.einsum('ijk,lj->ikl', self.W2, torch.concat((states[0], states[1]), 1))
            if self.layer == 1: inputstate2 = torch.einsum('li,ikl->lk',torch.concat((inputs[0], inputs[1]), 1), inputstate2_)
            else: inputstate2 = torch.einsum('li,ikl->lk',inputs, inputstate2_)
            u = self.sigmoid(inputstate2 + self.b2)
            # calculate new state
            new_state = u*state_tilda 
            new_state += (1.-u)*torch.einsum('ij,jk->ik', torch.concat((states[0], states[1]), 1), self.Wmerge)
        output = new_state
        return output, new_state





class Model(nn.Module):
    def __init__(self, input_size, system_size_x, system_size_y, N_target, sz_total, hidden_dim, weight_sharing, device, dtype=torch.float64):
        super(Model, self).__init__()
        """
        Creates RNN consisting of GRU cells.
        Inputs:
            - input_size:  number of quantum numbers (i.e. 2 for spin-1/2 particles)
            - system_size_x, _y: length of each snapshot in x,y direction
            - N_target: target number of particles in the lattice
            - sz_tot: target magnetization sector
            - hidden_dim:  dimension of hidden states
            - weight sharing: should be turned on, except for random states
            - device: CPU of GPU
        """
        # Defining some parameters
        self.N_target = N_target
        self.sz_tot = sz_total
        self.input_size  = input_size    # number of expected features in input data
        self.output_size = input_size    # number of expected features in output data
        self.N_x         = system_size_x # length of generated samples in x dir
        self.N_y         = system_size_y # length of generated samples in x dir
        self.hidden_dim  = hidden_dim    # number of features in the hidden state
        self.n_layer     = 1             # number of stacked GRUs, for Dilated RNN if will be set to 2 later on
        self.device      = device        # GPU or CPU 
        self.system_size = system_size_x*system_size_y
        self.weight_sharing = weight_sharing
        self.dtype       = dtype
        #Defining the layers
        if self.weight_sharing == True:
            print("Use RNN cell with weight sharing.")
            self.rnn  = TensorizedRNN(self.input_size, hidden_dim, self.device)   
            self.lin1 = nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=self.dtype)
            self.lin2 = nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=self.dtype)
        elif self.weight_sharing == False:
            self.rnn = []
            self.lin1 = []
            self.lin2 = []
            print("Use RNN cell without weight sharing.")
            for nl in range(self.n_layer):
                lin1 = []
                lin2 = []
                rnn = []
                for i in range(self.N_x):
                    for j in range(self.N_y):
                        if nl == 0:
                            lin1.append(nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=self.dtype))
                            lin2.append(nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=self.dtype))
                            rnn.append(TensorizedRNN(self.input_size, hidden_dim, self.device, layer=1))
                        else:
                            rnn.append(TensorizedRNN(hidden_dim, hidden_dim, self.device, layer=2))
                self.lin1.append(nn.ModuleList(lin1))
                self.lin2.append(nn.ModuleList(lin2))
                self.rnn.append(nn.ModuleList(rnn))
            self.lin1 = (nn.ModuleList(self.lin1))
            self.lin2 = (nn.ModuleList(self.lin2))
            self.rnn  = (nn.ModuleList(self.rnn))

        self.soft = nn.Softsign()
        
        self.get_num_parameters()


    def forward(self, samples):
        log_ampl, log_phase = self.log_probabilities(samples)
        return 0.5*log_ampl, log_phase

    def _forward(self, x, hidden, i, j):
        """
        Passes the input through the network.
        Inputs:
            - x:      input state at t
            - hidden: hidden state at t
        Outputs:
            - out:    output configuration at t+1
            - hidden: hidden state at t+1
        """
        # Passing in the input and hidden state into the model and obtaining outputs
        if self.weight_sharing:
            out, hidden = self.rnn(x[:2], hidden[:2])
            out = out.contiguous().view(-1, self.hidden_dim)
            hidden = [hidden]
        elif self.weight_sharing == False:
            out, hidden = self.rnn[0][i*self.N_y+j](x[:2], hidden[:2])
            out = out.contiguous().view(-1, self.hidden_dim)
            hidden = [hidden]
        return out, hidden

    def init_hidden(self, batch_size):
        """
        Generates the hidden state for a given batch size.
        """
        # This method generates the first hidden state of zeros for the forward pass and passes it to the device.
        # A vector of zeros would be  equivalent to a product state.
        hidden = torch.zeros((batch_size, self.hidden_dim), dtype=self.dtype, device=self.device)
        return hidden
    
    def get_num_parameters(self):
        """
        Calculates the number of parameters of the network. """
        p = 0
        print("Model parameters: ")
        for n, param in list(self.named_parameters()):
            if param.requires_grad:
                print(n+": "+str(param.numel()))
                p += param.numel()
        print("Total number of parameters in the network: "+str(p))
        self.num_params = p
        return p


    def enforce_N_total(self, num_up, num_dn, sampled_sites, amplitudes):
        """ Enforces the particle number and magnetization sectors as described in Hibat-Allah 2020 for the magnetization."""
        if self.input_size == 3:
            bl_up = ((self.N_target+self.sz_tot*2)//2)*torch.ones((amplitudes.size()[0],1)).to(self.device)
            bl_dn = ((self.N_target-self.sz_tot*2)//2)*torch.ones((amplitudes.size()[0],1)).to(self.device)
            bl_hole = (self.system_size-self.N_target)*torch.ones((amplitudes.size()[0],1)).to(self.device)
            ampl_up = torch.heaviside(bl_up-num_up, torch.tensor([0.], dtype=self.dtype, device=self.device))
            ampl_dn = torch.heaviside(bl_dn-num_dn, torch.tensor([0.], dtype=self.dtype, device=self.device))
            ampl_hole = torch.heaviside(bl_hole-(sampled_sites-(num_up+num_dn)), torch.tensor([0.], dtype=self.dtype, device=self.device))
            ampl = amplitudes * torch.stack([ampl_hole, ampl_dn, ampl_up], axis=1)[:,:,0]
            ampl = torch.nn.functional.normalize(ampl, p=1, eps = 1e-30)
        else:
            raise NotImplementedError("For input_dim != 3 not implemented")
        return ampl

    def _gen_samples(self, nx, ny, direction, inputs, hidden_inputs, samples, numsamples, num, num_up, num_dn, num_sites):
        # pass the hidden unit and sigma into the GRU cell at t=i 
        # and get the output y (will be used for calculating the 
        # probability) and the next hidden state
        full_sigma = [inputs[str(nx+direction[0])+str(ny)],inputs[str(nx)+str(ny+direction[1])]]
        hidden = []
        for nl in range(self.n_layer):
            hidden.append(hidden_inputs[nl][str(nx+(nl+1)*direction[0])+str(ny)])
            hidden.append(hidden_inputs[nl][str(nx)+str(ny+(nl+1)*direction[1])])

        y, hidden  = self._forward(full_sigma, hidden, nx, ny)
        # the amplitude is given by a linear layer with a softmax activation
        if not self.weight_sharing:
            ampl = self.lin1[0][nx*self.N_y+ny](y.to(self.dtype))
        else:
            ampl = self.lin1(y.to(self.dtype))
        ampl = torch.softmax(ampl,dim=1) # amplitude, all elements in a row sum to 1
        if self.sz_tot != None and self.N_target != None:
            ampl = self.enforce_N_total(num_up, num_dn, num_sites, ampl)
        # samples are obtained by sampling from the amplitudes
        sample = torch.multinomial(ampl, 1) 
        # calculate the number of sampled sites and spins
        sample_up = sample.clone()
        sample_up[sample==1] = 0
        sample_up[sample==2] = 1
        num_up += sample_up
        sample_dn = sample.clone()
        sample_dn[sample==2] = 0
        num_dn += sample_dn
        num_sites += 1
        # one hot encode the current sigma to pass it into the GRU at the next time step
        sigma = nn.functional.one_hot(sample, self.input_size).to(self.dtype).to(self.device)
        return sample[:,0], sigma, hidden, num_up, num_dn, num_sites
    
    
    def sample(self, num_samples):
        """
        Generates num_samples samples from the network and returns the samples,
        their log probabilities and phases.
        """
        # generate a first input of zeros (sigma and hidden states) to the first GRU cell at t=0
        sigma       = torch.zeros((num_samples,self.input_size), dtype=self.dtype, device=self.device)
        inputs = {}
        hidden_inputs = []
        for nl in range(self.n_layer):
            hidden_inputs.append({})
            for ny in range(-1-nl, self.N_y): # add a padding for the inputs and hidden states
                for nx in range(-1-nl, self.N_x+1+nl):
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[nl][str(nx)+str(ny)] = self.init_hidden(num_samples)
        
        samples     = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        num_up = torch.zeros((num_samples,1), dtype=self.dtype, device=self.device)
        num_dn = torch.zeros((num_samples,1), dtype=self.dtype, device=self.device)
        num_sites = torch.zeros((num_samples,1), dtype=self.dtype, device=self.device)
        num = 0
        for ny in range(self.N_y):
            if ny % 2 == 0: #go from left to right    
                for nx in range(self.N_x):
                    direction = [-1,-1]
                    samples[nx][ny], sigma, hidden, num_up, num_dn, num_sites = self._gen_samples(nx, ny, direction, inputs, hidden_inputs, samples, num_samples, num, num_up, num_dn, num_sites)
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[0][str(nx)+str(ny)]  = hidden[0]
                    if self.n_layer > 1: hidden_inputs[1][str(nx)+str(ny)]  = hidden[1]
                    num += 1
            else: #go from right to left
                for nx in range(self.N_x-1, -1, -1):
                    direction = [1,-1]
                    samples[nx][ny], sigma, hidden, num_up, num_dn, num_sites = self._gen_samples(nx, ny, direction, inputs, hidden_inputs, samples, num_samples, num, num_up, num_dn, num_sites)
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[0][str(nx)+str(ny)]  = hidden[0]
                    if self.n_layer > 1: hidden_inputs[1][str(nx)+str(ny)]  = hidden[1]
                    num += 1     
        samples = torch.stack([torch.stack(s, axis=1) for s in samples], axis=1).to(self.device)  
        return samples

    def _gen_probs(self, nx, ny, direction, samples, inputs, hidden_inputs, num, num_up, num_dn, num_sites):
        # pass the hidden unit and sigma into the GRU cell at t=i 
        # and get the output y (will be used for calculating the 
        # probability) and the next hidden state
        num_samples = samples.size()[0]
        sample = samples[:,nx,ny].reshape((samples.size()[0],1)) #index_select(index_select(samples, 1, torch.tensor([nx], device=self.device)), -1, torch.tensor([ny], device=self.device)).reshape((samples.size()[0],1))
        full_sigma = [inputs[str(nx+direction[0])+str(ny)],inputs[str(nx)+str(ny+direction[1])]]
        hidden = []
        for nl in range(self.n_layer):
            hidden.append(hidden_inputs[nl][str(nx+(nl+1)*direction[0])+str(ny)])
            hidden.append(hidden_inputs[nl][str(nx)+str(ny+(nl+1)*direction[1])])
        
        y, hidden  = self._forward(full_sigma, hidden, nx, ny)
        # the amplitude is given by a linear layer with a softmax activation
        if not self.weight_sharing:
            ampl = self.lin1[0][nx*self.N_y+ny](y)
        else:
            ampl = self.lin1(y)
        ampl = torch.softmax(ampl,dim=1) # amplitude, all elements in a row sum to 1
        if self.sz_tot != None and self.N_target != None:
            ampl = self.enforce_N_total(num_up, num_dn, num_sites, ampl)
        # the phase is given by a linear layer with a softsign activation
        if not self.weight_sharing:
            phase = self.lin2[0][nx*self.N_y+ny](y)
        else:
            phase = self.lin2(y)
        phase = self.soft(phase) 
        # calculate the number of sampled sites and spins
        sample_up = sample.clone()
        num_up = torch.where(sample_up==2,num_up+1,num_up)
        sample_dn = sample.clone()
        num_dn = torch.where(sample_dn==1,num_dn+1,num_dn)
        num_sites += 1
        # one hot encode the current sigma to pass it into the GRU at the next time step
        sigma = self.one_hot(sample, self.input_size).to(self.dtype) #.to(self.device)
        return sigma, ampl, torch.mul(torch.pi,phase), hidden, num_up, num_dn, num_sites
    
    def log_probabilities(self, samples):
        """
        Calculates the log probability and the phase of each item in samples.
        """
        # reshape samples
        samples = samples.to(self.device)
        if len(samples.size()) == 2:
            samples = torch.reshape(samples, (1,samples.size(0),samples.size(1)))
        num_samples = samples.size()[0]
        samples = samples.clone().detach()

        # generate a first input of zeros (sigma and hidden states) to the first GRU cell at t=0
        sigma  = torch.zeros((num_samples, self.input_size), dtype=self.dtype, device=self.device)
        inputs = {}
        hidden_inputs = []
        for nl in range(self.n_layer):
            hidden_inputs.append({})
            for ny in range(-1-nl, self.N_y):
                for nx in range(-1-nl, self.N_x+1+nl):
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[nl][str(nx)+str(ny)] = self.init_hidden(num_samples)
        num_up = torch.zeros((num_samples,1), dtype=self.dtype, device=self.device)
        num_dn = torch.zeros((num_samples,1), dtype=self.dtype, device=self.device)
        num_sites = torch.zeros((num_samples,1), dtype=self.dtype, device=self.device)
        ampl_probs  = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        phase_probs = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        ohs         = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        num = 0
        for ny in range(self.N_y):
            if ny % 2 == 0: #go from left to right
                for nx in range(self.N_x):
                    direction = [-1,-1]
                    sigma, ampl_probs[nx][ny], phase_probs[nx][ny], hidden, num_up, num_dn, num_sites = self._gen_probs(nx, ny, direction, samples, inputs, hidden_inputs, num, num_up, num_dn, num_sites)
                    ohs[nx][ny] = sigma.to(self.dtype)
                    inputs[str(nx)+str(ny)] = sigma.to(self.dtype)
                    hidden_inputs[0][str(nx)+str(ny)] = hidden[0]
                    if self.n_layer > 1: hidden_inputs[1][str(nx)+str(ny)] = hidden[1]
                    num += 1
            else: #go from right to left
                for nx in range(self.N_x-1, -1, -1):
                    direction = [1,-1]
                    sigma, ampl_probs[nx][ny], phase_probs[nx][ny], hidden, num_up, num_dn, num_sites = self._gen_probs(nx, ny, direction, samples, inputs, hidden_inputs, num, num_up, num_dn, num_sites)
                    ohs[nx][ny] = sigma.to(self.dtype)
                    inputs[str(nx)+str(ny)] = sigma.to(self.dtype)
                    hidden_inputs[0][str(nx)+str(ny)] = hidden[0]
                    if self.n_layer > 1: hidden_inputs[1][str(nx)+str(ny)] = hidden[1]
                    num += 1
        ampl_probs = torch.cat([torch.stack(a, axis=1) for a in ampl_probs], axis=1) 
        phase_probs = torch.cat([torch.stack(p, axis=1) for p in phase_probs], axis=1) 
        ohs = torch.cat([torch.cat(o, axis=1) for o in ohs], axis=1)
        # calculate the wavefunction and split it into amplitude and phase
        log_probs_ampl = torch.sum(torch.log(torch.sum(torch.torch.multiply(ampl_probs,ohs), axis =2)), axis=1)
        phase = torch.sum((torch.sum(torch.torch.multiply(phase_probs,ohs), axis =2)), axis=1)
        return log_probs_ampl, phase

    def one_hot(self, indices, num_classes):
        values = torch.arange(num_classes, device=indices.device)

        # Create a tensor of shape (indices.size(0), 1) containing the class indices
        class_indices = indices.unsqueeze(1)

        # Use the torch.eq function to create a boolean mask indicating
        # where the class indices match the row indices
        mask = torch.eq(class_indices, values)
        # Use the mask to set the appropriate values in the one_hot tensor to 1
        one_hot = torch.where(mask==True, 1,0)
        return one_hot

