import torch
from torch import nn
import numpy as np
import timeit

torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(1234)

class TensorizedRNN(nn.Module):
    """ Custom RNN / GRU layer for 2D input """
    def __init__(self, input_size, hidden_size, device, layer=1, celltype="GRU"):
        super().__init__()
        
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.sigmoid  = torch.nn.Sigmoid()
        self.tanh     = torch.nn.Tanh()
        self.elu      = torch.nn.ELU()
        self.device   = device
        self.celltype = celltype        
        self.layer = layer

        # define all weights
        if self.layer == 1:
            input_dim = 2*self.input_size
        else:
            input_dim = self.input_size
        factory_kwargs = {'device': device, 'dtype': torch.float32}
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
            #fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.W2)
            #lim = np.sqrt(3.0 / (0.5*(fan_in+fan_out)))
            #nn.init.uniform_(self.b2, -lim, lim)
            self.b2.data.fill_(0.01)
        self.b1.data.fill_(0.01)
        #fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.W1)
        #lim = np.sqrt(3.0 / (0.5*(fan_in+fan_out)))
        #nn.init.uniform_(self.b1, -lim, lim)


    def forward(self, inputs, states):
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
            # calculate h_tilde in Eq. A1
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
    def __init__(self, input_size, system_size_x, system_size_y, N_target, sz_total, hidden_dim, weight_sharing, device):
        super(Model, self).__init__()
        """
        Creates RNN consisting of GRU cells.
        Inputs:
            - input_size:  number of quantum numbers (i.e. 2 for spin-1/2 particles)
            - system_size: length of each snapshot
            - hidden_dim:  dimension of hidden states
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
        self.device      = device 
        self.rnn_type    = "Non-Dilated"
        self.system_size = system_size_x*system_size_y
        self.weight_sharing = weight_sharing
        #Defining the layers
        if self.weight_sharing == True:
            print("Use RNN cell with weight sharing.")
            self.rnn  = TensorizedRNN(self.input_size, hidden_dim, self.device)   
            self.lin1 = nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=torch.float32)
            self.lin2 = nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=torch.float32)
        elif self.weight_sharing == False or self.rnn_type == "Dilated":
            self.rnn = []
            self.lin1 = []
            self.lin2 = []
            if self.rnn_type == "Dilated": 
                print("Use Dilated RNN cell without weight sharing.")
                self.n_layer = 2
            print("Use RNN cell without weight sharing.")
            for nl in range(self.n_layer):
                lin1 = []
                lin2 = []
                rnn = []
                for i in range(self.N_x):
                    for j in range(self.N_y):
                        if nl == 0:
                            lin1.append(nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=torch.float32))
                            lin2.append(nn.Linear(hidden_dim, self.output_size, device=self.device, dtype=torch.float32))
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
 
    def forward(self, x, hidden, i, j):
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
        elif self.weight_sharing == False and self.rnn_type != "Dilated":
            out, hidden = self.rnn[0][i*self.N_y+j](x[:2], hidden[:2])
            out = out.contiguous().view(-1, self.hidden_dim)
            hidden = [hidden]
        if self.rnn_type=="Dilated":
            o, h1 = self.rnn[0][i*self.N_y+j](x[:2], hidden[:2])
            out, h2 = self.rnn[1][i*self.N_y+j](o, hidden[2:4])
            out = out.contiguous().view(-1, self.hidden_dim)
            hidden = [h1, h2]
        return out, hidden
    
    def init_hidden(self, batch_size):
        """
        Generates the hidden state for a given batch size.
        """
        # This method generates the first hidden state of zeros for the forward pass and passes it to the device.
        # A vector of zeros would be  equivalent to a product state.
        hidden = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float32).to(self.device)
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
        return p

    def enforce_N_total(self, num_up, num_dn, sampled_sites, amplitudes):
        if self.input_size == 3:
            bl_up = ((self.N_target+self.sz_tot)//2)*torch.ones((amplitudes.size()[0],1)).to(self.device)
            bl_dn = ((self.N_target-self.sz_tot)//2)*torch.ones((amplitudes.size()[0],1)).to(self.device)
            bl_hole = (self.system_size-self.N_target)*torch.ones((amplitudes.size()[0],1)).to(self.device)
            ampl_up = torch.heaviside(bl_up-num_up, torch.tensor([0.]).to(self.device))
            ampl_dn = torch.heaviside(bl_dn-num_dn, torch.tensor([0.]).to(self.device))
            ampl_hole = torch.heaviside(bl_hole-(sampled_sites-(num_up+num_dn)), torch.tensor([0.]).to(self.device))
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

        y, hidden  = self.forward(full_sigma, hidden, nx, ny)
        # the amplitude is given by a linear layer with a softmax activation
        if not self.weight_sharing or self.rnn_type=="Dilated":
            ampl = self.lin1[0][nx*self.N_y+ny](y.to(torch.float32))
        else:
            ampl = self.lin1(y.to(torch.float32))
        ampl = torch.softmax(ampl,dim=1) # amplitude, all elements in a row sum to 1
        if self.sz_tot != None and self.N_target != None:
            ampl = self.enforce_N_total(num_up, num_dn, num_sites, ampl)
        # samples are obtained by sampling from the amplitudes
        sample = torch.multinomial(ampl, 1) 
        # one hot encode the current sigma to pass it into the GRU at
        # the next time step
        sample_up = sample.clone()
        sample_up[sample==1] = 0
        sample_up[sample==2] = 1
        num_up += sample_up
        sample_dn = sample.clone()
        sample_dn[sample==2] = 0
        num_dn += sample_dn
        num_sites += 1
        sigma = nn.functional.one_hot(sample, self.input_size).to(torch.float32).to(self.device)
        return sample[:,0], sigma, hidden, num_up, num_dn, num_sites
    
    
    def sample(self, num_samples):
        """
        Generates num_samples samples from the network and returns the samples,
        their log probabilities and phases.
        """
        # generate a first input of zeros (sigma and hidden states) to the first GRU cell at t=0
        sigma       = torch.zeros((num_samples,self.input_size), dtype=torch.float32).to(self.device)
        inputs = {}
        hidden_inputs = []
        for nl in range(self.n_layer):
            hidden_inputs.append({})
            for ny in range(-1-nl, self.N_y): # add a padding for the inputs and hidden states
                for nx in range(-1-nl, self.N_x+1+nl):
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[nl][str(nx)+str(ny)] = self.init_hidden(num_samples)
        
        samples     = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        num_up = torch.zeros((num_samples,1), dtype=torch.float32).to(self.device)
        num_dn = torch.zeros((num_samples,1), dtype=torch.float32).to(self.device)
        num_sites = torch.zeros((num_samples,1), dtype=torch.float32).to(self.device)
        num = 0
        #start = timeit.default_timer()
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
        samples = torch.stack([torch.stack(s, axis=1) for s in samples], axis=1).to(self.device)  #.reshape((num_samples, self.N_x, self.N_y))
        return samples
    
    def _gen_probs(self, nx, ny, direction, samples, inputs, hidden_inputs, num, num_up, num_dn, num_sites):
        # pass the hidden unit and sigma into the GRU cell at t=i 
        # and get the output y (will be used for calculating the 
        # probability) and the next hidden state
        sample = samples[:,nx,ny].reshape((samples.size()[0],1))
        full_sigma = [inputs[str(nx+direction[0])+str(ny)],inputs[str(nx)+str(ny+direction[1])]]
        hidden = []
        for nl in range(self.n_layer):
            hidden.append(hidden_inputs[nl][str(nx+(nl+1)*direction[0])+str(ny)])
            hidden.append(hidden_inputs[nl][str(nx)+str(ny+(nl+1)*direction[1])])
        
        y, hidden  = self.forward(full_sigma, hidden, nx, ny)
        # the amplitude is given by a linear layer with a softmax activation
        if not self.weight_sharing or self.rnn_type=="Dilated":
            ampl = self.lin1[0][nx*self.N_y+ny](y)
        else:
            ampl = self.lin1(y)
        ampl = torch.softmax(ampl,dim=1) # amplitude, all elements in a row sum to 1
        if self.sz_tot != None and self.N_target != None:
            ampl = self.enforce_N_total(num_up, num_dn, num_sites, ampl)
        # the phase is given by a linear layer with a softsign activation
        if not self.weight_sharing or self.rnn_type=="Dilated":
            phase = self.lin2[0][nx*self.N_y+ny](y)
        else:
            phase = self.lin2(y)
        phase = self.soft(phase) 
        # one hot encode the current sigma to pass it into the GRU at
        # the next time step
        sample_up = sample.clone()
        sample_up[sample==1] = 0
        sample_up[sample==2] = 1
        num_up += sample_up
        sample_dn = sample.clone()
        sample_dn[sample==2] = 0
        num_dn += sample_dn
        num_sites += 1

        sigma = nn.functional.one_hot(sample.reshape((sample.size()[0],1)), self.input_size).to(torch.float32).to(self.device)
        
        return sigma, ampl, torch.mul(torch.pi,phase), hidden, num_up, num_dn, num_sites
    
    def log_probabilities(self, samples):
        """
        Calculates the log probability and the phase of each item in samples.
        """
        # reshape samples
        samples = samples.to(self.device)
        num_samples = samples.size()[0]
        samples = samples.clone().detach()
        
        # generate a first input of zeros (sigma and hidden states) to the first GRU cell at t=0
        sigma  = torch.zeros((num_samples, self.input_size), dtype=torch.float32).to(self.device)
        inputs = {}
        hidden_inputs = []
        for nl in range(self.n_layer):
            hidden_inputs.append({})
            for ny in range(-1-nl, self.N_y):
                for nx in range(-1-nl, self.N_x+1+nl):
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[nl][str(nx)+str(ny)] = self.init_hidden(num_samples)
        num_up = torch.zeros((num_samples,1), dtype=torch.float32).to(self.device)
        num_dn = torch.zeros((num_samples,1), dtype=torch.float32).to(self.device)
        num_sites = torch.zeros((num_samples,1), dtype=torch.float32).to(self.device)
        ampl_probs  = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        phase_probs = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        ohs         = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        num = 0
        #start = timeit.default_timer()
        for ny in range(self.N_y):
            if ny % 2 == 0: #go from left to right
                for nx in range(self.N_x):
                    direction = [-1,-1]
                    sigma, ampl_probs[nx][ny], phase_probs[nx][ny], hidden, num_up, num_dn, num_sites = self._gen_probs(nx, ny, direction, samples, inputs, hidden_inputs, num, num_up, num_dn, num_sites)
                    ohs[nx][ny] = sigma
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[0][str(nx)+str(ny)] = hidden[0]
                    if self.n_layer > 1: hidden_inputs[1][str(nx)+str(ny)]  = hidden[1]
                    num += 1
            else: #go from right to left
                for nx in range(self.N_x-1, -1, -1):
                    direction = [1,-1]
                    sigma, ampl_probs[nx][ny], phase_probs[nx][ny], hidden, num_up, num_dn, num_sites = self._gen_probs(nx, ny, direction, samples, inputs, hidden_inputs, num, num_up, num_dn, num_sites)
                    ohs[nx][ny] = sigma
                    inputs[str(nx)+str(ny)] = sigma
                    hidden_inputs[0][str(nx)+str(ny)] = hidden[0]
                    if self.n_layer > 1: hidden_inputs[1][str(nx)+str(ny)]  = hidden[1]
                    num += 1
        #end = timeit.default_timer()
        ampl_probs = torch.cat([torch.stack(a, axis=1) for a in ampl_probs], axis=1) #.reshape((num_samples, self.N_x*self.N_y, 2))
        phase_probs = torch.cat([torch.stack(p, axis=1) for p in phase_probs], axis=1) #.reshape((num_samples, self.N_x*self.N_y, 2))
        ohs = torch.cat([torch.cat(o, axis=1) for o in ohs], axis=1) #.reshape((num_samples, self.N_x*self.N_y, 2))
        # calculate the wavefunction and split it into amplitude and phase
        log_probs_ampl = torch.sum(torch.log(torch.sum(torch.torch.multiply(ampl_probs,ohs), axis =2)), axis=1)
        phase = torch.sum((torch.sum(torch.torch.multiply(phase_probs,ohs), axis =2)), axis=1)
        return log_probs_ampl, phase
        
