import os
import time

import numpy as np
import torch
from torch import Tensor
from torch import nn
from model import Model

from functorch import jacrev, vmap, make_functional, grad



def compute_centered_jacobian(model,samples,batchsize=2000,make_sparse=True):
    # Computes real and imaginary parts of O=d Psi/d Theta. 
    func, parameters = make_functional(model)
    jac_ampl = []
    jac_phase = []
    # calculate the jacobian of the logarithmic wave function
    for batch in range(max(1,int(samples.size(0)/batchsize))):
        batch_jac = vmap(jacrev(func), in_dims=(None,0))(parameters, samples[batch*batchsize:batchsize*(1+batch)])
        ja = batch_jac[0]
        ja = torch.cat([it.reshape(it.size(0),-1) for it in list(ja)], axis=-1)
        jp = batch_jac[1]
        jp = torch.cat([it.reshape(it.size(0),-1) for it in list(jp)], axis=-1)
        jac_ampl.append(ja)
        jac_phase.append(jp)
        del jp, ja, batch_jac
    jac_ampl = torch.cat(jac_ampl, axis=0)
    jac_phase = torch.cat(jac_phase, axis=0)
    # subtract the mean
    with torch.no_grad():
        jac_ampl  -= jac_ampl.mean(axis=0)
        jac_phase -= jac_phase.mean(axis=0)
    jac = torch.cat([jac_ampl,jac_phase], axis=0)
    if make_sparse: # remove zero values
        non_zero_columns = torch.where(jac.sum(axis=0) != 0)[0]
        jac = jac[:,jac.sum(axis=0)!=0]
        assert jac.size(1) == non_zero_columns.size(0) 
    else:
        non_zero_columns = None
    jac = jac / (np.sqrt(jac_ampl.size(0)))
    return jac, non_zero_columns


def invert(M, X, f, rtol=1e-14, onDevice=True, snr_regularization=True):
    if onDevice:
        ev, V = torch.linalg.eigh(M)
    else:
        ev, V = np.linalg.eigh(M.cpu().detach().numpy())
        ev = torch.tensor(ev, device=M.device)
        V  = torch.tensor(V, device=M.device)
    Vt = torch.conj(V).t()
    invEv = torch.where(torch.abs(ev / ev[-1]) > rtol, 1. / ev, 0.)
    Minv = torch.einsum("ij,jk,k", X.t(), torch.diag(invEv),f)
    return Minv


def compute_gradient_with_curvature(X, f, model, epsilon):
    M = torch.einsum("ij,jk", X,X.t()) + epsilon * torch.diag(torch.ones(X.size(0))).to(X.device)
    δ = invert(M, X, f) #.to(torch.complex128)
    print("delta", torch.linalg.norm(δ), torch.max(δ))
    return δ

def apply_grads(model,sparse_grad,non_zero_columns):
    """ Assigns the calculated gradients to the model parameter gradients. """
    i = 0
    # insert zeros again
    num_params = model.num_params
    if non_zero_columns.size(0)!=0:
        grad = torch.zeros(num_params,dtype=torch.float64,device=model.device)
        grad[non_zero_columns] = sparse_grad
    else:
        grad = sparse_grad
    for p in filter(lambda x: x.requires_grad, model.parameters()):
        n = p.numel()
        assert p.is_leaf
        if p.grad is not None:
            p.grad.copy_(grad[i : i + n].view(p.size()))
        else:
            print("gradient = None. Please check whats going wrong!")
            p.grad = grad[i : i + n].view(p.size())
        i += n


def run_sr(model, E, samples, optimizer, epsilon=1e-4, scheduler=None):
    """ Runs a minSR step. """
    # Campute the real and imaginary part of the jacobian matrix. 
    X, indices = compute_centered_jacobian(model, samples)
    E = -((E-E.mean()) / np.sqrt(E.size(0)))
    f = torch.cat([E.real, -E.imag])
    # Compute the gradients of the NN parameters from that.
    sparse_grads = compute_gradient_with_curvature(X, f, model, epsilon)
    # Assign the calculated gradients and make one optimization step.
    cost = torch.linalg.norm(sparse_grads).detach().cpu().numpy()
    if cost >= 1000:
        print("Gradient = "+str(cost)+" --> Cut gradient.")
        sparse_grads *= 1e-8/cost
    apply_grads(model,sparse_grads,indices)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return cost, torch.einsum("ij,jk", X,X.t())
