import os
import time

import numpy as np
import torch
from torch import Tensor
from torch import nn
from model import Model




def autograd(input, params):
    return torch.autograd.grad(input, params, allow_unused=True, retain_graph=True)

def _compute_centered_jacobian(model,samples):
    """ Computes O=d Psi/d Theta. """
    l = tuple(filter(lambda p: p.requires_grad, model.parameters()))
    parameters = l
    jac = []
    log_probs, phases = model.log_probabilities(samples)
    weights = torch.exp(log_probs)
    jac_ampl = [torch.cat([j_.flatten() for j_ in autograd(0.5*log_probs[i], parameters) if j_!=None]) for i in range(log_probs.size(0))]
    jac_phase = [torch.cat([j_.flatten() for j_ in autograd(phases[i], parameters) if j_!= None]) for i in range(phases.size(0))]
    jac_ampl = torch.stack(jac_ampl) #/ np.sqrt(samples.size(0)) 
    jac_phase = torch.stack(jac_phase) #/ np.sqrt(samples.size(0))
    jac_ampl -= weights @ jac_ampl
    jac_phase -= weights @ jac_phase
    return jac_ampl, jac_phase, log_probs, phases



def compute_centered_jacobian(model, samples):
    Ore, Oim, log_probs, phases = _compute_centered_jacobian(model,samples)
    return Ore, Oim, log_probs, phases


def _compute_T(Ore, Oim):
    if Ore is None:
        return None
    T = torch.einsum("ij,jk", Ore, Ore.t())+torch.einsum("ij,jk",Oim,Oim.t())  
    Tinv = torch.linalg.pinv(T,rtol=1e-12)
    return T

def _compute_gradient_with_curvature(Tinv, E, O, weights):
    TinvE = torch.mv(Tinv, E)
    δ = torch.einsum("ij,j,j", O.t(),TinvE,weights)
    return δ


def compute_gradient_with_curvature(Ore, Oim, E, weights, model,**kwargs):
    # The following is an implementation of Eq. (6.22) (without the minus sign) in "Quantum Monte
    # Carlo Approaches for Correlated Systems" by F.Becca & S.Sorella.
    #
    # Basically, what we need to compute is `2/N · Re[E*·(O - ⟨O⟩)]`, where `E` is a `Nx1` vector
    # of complex-numbered local energies, `O` is a `NxM` matrix of logarithmic derivatives, `⟨O⟩` is
    # a `1xM` vector of mean logarithmic derivatives, and `N` is the number of Monte Carlo samples.
    #
    # Now, `Re[a*·b] = Re[a]·Re[b] - Im[a*]·Im[b] = Re[a]·Re[b] +
    # Im[a]·Im[b]` that's why there are no conjugates or minus signs in
    # the code.
    E = (E-E.mean()) * weights
    Tinv = _compute_T(Ore, Oim)
    δ = _compute_gradient_with_curvature(Tinv, E.real, Ore, weights)+_compute_gradient_with_curvature(Tinv, E.imag, Oim, weights)
    return δ

def apply_grads(model,grad):
    i = 0
    for p in filter(lambda x: x.requires_grad, model.parameters()):
        n = p.numel()
        if p.grad is not None:
            p.grad.copy_(grad[i : i + n].view(p.size()))
        else:
            print("gradient = None. Please check whats going wrong!")
            p.grad = grad[i : i + n].view(p.size())
        i += 1

def run_sr(model, E, samples, optimizer, scheduler=None):
    Ore, Oim, log_probs, phases = compute_centered_jacobian(model, samples)
    weights = torch.exp(log_probs)
    grads = compute_gradient_with_curvature(Oim, Ore, E, weights, model)
    print(grads)
    apply_grads(model,grads)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
