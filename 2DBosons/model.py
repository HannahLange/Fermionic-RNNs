import torch
from torch import nn
import numpy as np


class TensorizedGRU(nn.Module):
    """ Custom GRU layer for 2D input """
    def __init__(self, input_size, hidden_size, device):
        super().__init__()
        
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh    = torch.nn.Tanh()
        self.device  = device
        
          
        # define all weights
        w1      = torch.empty(self.hidden_size, 2*self.hidden_size, 2*self.input_size)
        self.W1 = nn.Parameter(w1).to(self.device)  # nn.Parameter is a Tensor that's a module parameter.
        b1      = torch.empty(self.hidden_size)
        self.b1 = nn.Parameter(b1).to(self.device)
        
        w2      = torch.empty(self.hidden_size, 2*self.hidden_size, 2*self.input_size)
        self.W2 = nn.Parameter(w2).to(self.device) 
        b2      = torch.empty(self.hidden_size)
        self.b2 = nn.Parameter(b2).to(self.device)
        
        w3      = torch.empty(2*self.hidden_size, self.hidden_size)
        self.W3 = nn.Parameter(w3).to(self.device) 

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1, 1)
        nn.init.xavier_uniform_(self.W2, 1)
        nn.init.xavier_uniform_(self.W3, 1)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.W1)
        lim = np.sqrt(3.0 / (0.5*(fan_in+fan_out)))
        nn.init.uniform_(self.b1, -lim, lim)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.W2) 
        lim = np.sqrt(3.0 / (0.5*(fan_in+fan_out)))
        nn.init.uniform_(self.b2, -lim, lim)
   

    def forward(self, inputs, states):
        if len(inputs[0].size()) == 3:
            inputs[0] = inputs[0][:,0,:]
        if len(inputs[1].size()) == 3:
            inputs[1] = inputs[1][:,0,:]

        inputstate_mul = torch.einsum('ij,ik->ijk', torch.concat((states[0], states[1]), 1),torch.concat((inputs[0], inputs[1]),1))
        # prepare input linear combination
        state_mul1 = torch.einsum('ijk,ljk->il', inputstate_mul, self.W1) # [batch_sz, num_units]
        state_mul2 = torch.einsum('ijk,ljk->il', inputstate_mul, self.W2) # [batch_sz, num_units]

        u = self.sigmoid(state_mul2 + self.b2)
        state_tilda = self.tanh(state_mul1 + self.b1) 

        new_state = u*state_tilda 
        new_state += (1.-u)*torch.einsum('ij,jk->ik', torch.concat((states[0], states[1]), 1), self.W3)
        output = new_state
        return output, new_state



class Model(nn.Module):
    def __init__(self, input_size, system_size_x, system_size_y, hidden_dim, n_layers, device):
        super(Model, self).__init__()
        """
        Creates RNN consisting of GRU cells.
        Inputs:
            - input_size:  number of quantum numbers (i.e. 2 for spin-1/2 particles)
            - system_size: length of each snapshot
            - hidden_dim:  dimension of hidden states
            - n_layers:    number of layers of the GRU
        """

        # Defining some parameters
        self.input_size  = input_size    # number of expected features in input data
        self.output_size = input_size    # number of expected features in output data
        self.N_x         = system_size_x # length of generated samples in x dir
        self.N_y         = system_size_y # length of generated samples in x dir
        self.hidden_dim  = hidden_dim    # number of features in the hidden state
        self.n_layers    = n_layers      # number of stacked GRUs
        self.device      = device 
        self.system_size = system_size_x*system_size_y
        #Defining the layers
        self.rnn  = TensorizedGRU(self.input_size, hidden_dim, self.device)   
        self.lin1 = nn.Linear(hidden_dim, self.output_size)
        self.lin2 = nn.Linear(hidden_dim, self.output_size)
        #self.s    = torch.softmax(dim=0)
        self.soft = nn.Softsign()
        
        self.get_num_parameters()
        
    def forward(self, x, hidden):
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
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the dense layer
        out = out.contiguous().view(-1, self.hidden_dim)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """
        Generates the hidden state for a given batch size.
        """
        # This method generates the first hidden state of zeros for the forward pass and passes it to the device.
        # This is equivalent to a product state.
        hidden = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float64).to(self.device)
        return hidden
    
    def get_num_parameters(self):
        """
        Calculates the number of parameters of the network. """
        p = 0
        for param in list(self.parameters()):
            if param.requires_grad:
                p += param.numel()
        print("Total number of parameters in the network: "+str(p))
        return p
    
    def _gen_samples(self, nx, ny, direction, inputs, hidden_inputs, numsamples):
        # pass the hidden unit and sigma into the GRU cell at t=i 
        # and get the output y (will be used for calculating the 
        # probability) and the next hidden state
        full_sigma = [inputs[str(nx+direction[0])+str(ny)],inputs[str(nx)+str(ny+direction[1])]]
        hidden     = [hidden_inputs[str(nx+direction[0])+str(ny)],hidden_inputs[str(nx)+str(ny+direction[1])]]
        y, hidden  = self.forward(full_sigma, hidden)
        # the amplitude is given by a linear layer with a softmax activation
        ampl = self.lin1(y)
        ampl = torch.softmax(ampl,dim=1) # amplitude, all elements in a row sum to 1
        # samples are obtained by sampling from the amplitudes
        sample = torch.multinomial(ampl, 1) 
        # one hot encode the current sigma to pass it into the GRU at
        # the next time step
        sigma = nn.functional.one_hot(sample, 2).double()
        
        return sample[:,0], sigma, hidden
    
    
    def sample(self, num_samples):
        """
        Generates num_samples samples from the network and returns the samples,
        their log probabilities and phases.
        """
        # generate a first input of zeros (sigma and hidden states) to the first GRU cell at t=0
        sigma       = torch.zeros((num_samples,2), dtype=torch.float64).to(self.device)
        inputs = {}
        hidden_inputs = {}
        for ny in range(-1, self.N_y): # add a padding for the inputs and hidden states
            for nx in range(-1, self.N_x+1):
                inputs[str(nx)+str(ny)] = sigma
                hidden_inputs[str(nx)+str(ny)] = self.init_hidden(num_samples)
        
        samples = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        for ny in range(self.N_y):
            if ny % 2 == 0: #go from left to right
                for nx in range(self.N_x):
                    direction = [-1,-1]
                    samples[nx][ny], sigma, hidden_inputs[str(nx)+str(ny)] = self._gen_samples(nx, ny, direction, inputs, hidden_inputs, num_samples)
                    inputs[str(nx)+str(ny)] = sigma
            else: #go from right to left
                for nx in range(self.N_x-1, -1, -1):
                    direction = [1,-1]
                    samples[nx][ny], sigma, hidden_inputs[str(nx)+str(ny)] = self._gen_samples(nx, ny, direction, inputs, hidden_inputs, num_samples)
                    inputs[str(nx)+str(ny)] = sigma
                    
        samples = torch.stack([torch.stack(s, axis=1) for s in samples], axis=1).to(self.device)  #.reshape((num_samples, self.N_x, self.N_y))
        return samples
    
    def _gen_probs(self, nx, ny, direction, sample, inputs, hidden_inputs):
        # pass the hidden unit and sigma into the GRU cell at t=i 
        # and get the output y (will be used for calculating the 
        # probability) and the next hidden state
        full_sigma = [inputs[str(nx+direction[0])+str(ny)],inputs[str(nx)+str(ny+direction[1])]]
        hidden     = [hidden_inputs[str(nx+direction[0])+str(ny)],hidden_inputs[str(nx)+str(ny+direction[1])]]
        y, hidden  = self.forward(full_sigma, hidden)
        # the amplitude is given by a linear layer with a softmax activation
        ampl = self.lin1(y)
        ampl = torch.softmax(ampl,dim=1) # amplitude, all elements in a row sum to 1
        # the phase is given by a linear layer with a softsign activation
        phase = self.lin2(y)
        phase = self.soft(phase) 
        # one hot encode the current sigma to pass it into the GRU at
        # the next time step
        sigma = nn.functional.one_hot(sample.reshape((sample.size()[0],1)), 2).double()
        
        return sigma, ampl, torch.mul(torch.pi,phase), hidden
    
    def log_probabilities(self, samples):
        """
        Calculates the log probability and the phase of each item in samples.
        """
        # reshape samples
        samples = samples.to(self.device)
        num_samples = samples.size()[0]
        samples = samples.clone().detach()
        
        # generate a first input of zeros (sigma and hidden states) to the first GRU cell at t=0
        sigma  = torch.zeros((num_samples,2), dtype=torch.float64).to(self.device)
        inputs = {}
        hidden_inputs = {}
        for ny in range(-1, self.N_y):
            for nx in range(-1, self.N_x+1):
                inputs[str(nx)+str(ny)] = sigma
                hidden_inputs[str(nx)+str(ny)] = self.init_hidden(num_samples)

        ampl_probs  = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        phase_probs = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        ohs         = [[[] for ny in range(self.N_y)] for nx in range(self.N_x)]
        for ny in range(self.N_y):
            if ny % 2 == 0: #go from left to right
                for nx in range(self.N_x):
                    direction = [-1,-1]
                    sigma, ampl_probs[nx][ny], phase_probs[nx][ny], hidden_inputs[str(nx)+str(ny)] = self._gen_probs(nx, ny, direction, samples[:,nx,ny], inputs, hidden_inputs)
                    ohs[nx][ny] = sigma
                    inputs[str(nx)+str(ny)] = sigma
            else: #go from right to left
                for nx in range(self.N_x-1, -1, -1):
                    direction = [1,-1]
                    sigma, ampl_probs[nx][ny], phase_probs[nx][ny], hidden_inputs[str(nx)+str(ny)] = self._gen_probs(nx, ny, direction, samples[:,nx,ny], inputs, hidden_inputs)
                    ohs[nx][ny] = sigma
                    inputs[str(nx)+str(ny)] = sigma
        ampl_probs = torch.cat([torch.stack(a, axis=1) for a in ampl_probs], axis=1) #.reshape((num_samples, self.N_x*self.N_y, 2))
        phase_probs = torch.cat([torch.stack(p, axis=1) for p in phase_probs], axis=1) #.reshape((num_samples, self.N_x*self.N_y, 2))
        ohs = torch.cat([torch.cat(o, axis=1) for o in ohs], axis=1) #.reshape((num_samples, self.N_x*self.N_y, 2))
        # calculate the wavefunction and split it into amplitude and phase
        log_probs_ampl = torch.sum(torch.log(torch.sum(torch.torch.multiply(ampl_probs,ohs), axis =2)), axis=1)
        phase = torch.sum((torch.sum(torch.torch.multiply(phase_probs,ohs), axis =2)), axis=1)
        return log_probs_ampl, phase
        
