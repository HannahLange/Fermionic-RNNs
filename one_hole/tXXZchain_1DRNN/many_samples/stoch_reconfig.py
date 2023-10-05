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
        jac_ampl = jac_ampl[:,jac.sum(axis=0)!=0]
        jac_phase = jac_phase[:,jac.sum(axis=0)!=0]
        assert jac_ampl.size(1) == non_zero_columns.size(0) and jac_phase.size(1) == non_zero_columns.size(0)
    else:
        non_zero_columns = None
    jac_ampl = jac_ampl / (torch.linalg.norm(jac_ampl)*np.sqrt(jac_ampl.size(0)))
    jac_phase =  jac_phase / (torch.linalg.norm(jac_phase)*np.sqrt(jac_phase.size(0)))
    print(non_zero_columns.size(0))
    return jac_ampl, jac_phase, non_zero_columns


def invert(M, O, E, rtol=1e-14, onDevice=True, snr_regularization=True):
    if onDevice:
        ev, V = torch.linalg.eigh(M)
    else:
        ev, V = np.linalg.eigh(M.cpu().detach().numpy())
        ev = torch.tensor(ev, device=M.device)
        V  = torch.tensor(V, device=M.device)
    Vt = torch.conj(V).t()
    invEv = torch.where(torch.abs(ev / ev[-1]) > rtol, 1. / ev, 0.)
    """
    regularizer = 1. / (1. + (rtol * torch.abs(ev[-1]) / torch.abs(ev))**6) 
    if snr_regularization:
        U, _, _ = torch.svd(O)
        rho_p = torch.mv(torch.conj(U).t(), E).to(torch.float64)
        denom = torch.sqrt((torch.mv(Vt,E).to(torch.float64)-rho_p)**2)/rho_p.size(0)**2
        snr = (rho_p/denom)
    print("invEv", torch.linalg.norm(invEv))
    regularizer = 1. / (1. + (rtol * torch.abs(ev[-1]) / torch.abs(ev))**6) 
    print("reg", torch.linalg.norm(regularizer))
    regularizer *= rho_p / (1. + (2 / snr)**6)
    print("reg", torch.linalg.norm(regularizer))
    """
    Minv = torch.einsum("ij,jk,kl", V, torch.diag(invEv).to(torch.complex128),Vt)
    print((torch.abs(torch.matmul(Minv, M)-torch.diag(torch.ones(M.size(0),dtype=torch.float64,device=M.device))).sum()/(M.size(0)*M.size(1))).detach().cpu().numpy())
    return Minv


def compute_gradient_with_curvature(Ore, Oim, E, model, epsilon):
    E = -((E-E.mean()) / np.sqrt(E.size(0)))
    O = torch.complex(Ore, Oim)
    print("O", torch.linalg.norm(O), torch.max(O.float()), torch.min(O.float()))
    Tre = torch.matmul(Ore, Ore.t()) + torch.matmul(Oim, Oim.t())
    Tim = torch.matmul(Oim, Ore.t()) - torch.matmul(Ore, Oim.t())
    T = torch.complex(Tre, Tim)
    T_ = T + torch.diag(torch.ones(T.size(0), device=T.device, dtype=torch.complex128)*epsilon)  #torch.diag(1e-4*torch.diag(T.real)) 
    print(torch.linalg.norm(T), torch.max(torch.abs(T.float())), torch.min(torch.abs(T.float())))
    #T_im = Tim + torch.diag(torch.ones(Tim.size(0), device=Tim.device, dtype=torch.float64)*epsilon)
    Tinv = invert(T_, O, E) #.to(torch.complex128)
    print("Tinv", torch.linalg.norm(Tinv), torch.max(torch.abs(Tinv.float())), torch.min(torch.abs(Tinv.float())))
    #Tinv_re = invert(Tre+torch.einsum("ij,jk,kl", Tim,Treinv,Tim))
    #Tinv_im = -torch.einsum("ij,jk,kl", Treinv,Tim,Tinv_re)
    #Tinv = torch.complex(Tinv_re, Tinv_im)
    TinvE = torch.mv(Tinv, E)
    δ = torch.real(torch.mv(torch.conj(O).t(),TinvE)).to(torch.float64)
    print("delta", torch.linalg.norm(δ), torch.max(δ))
    inversion_error = torch.real(torch.linalg.norm((torch.mv(T.float(),TinvE.float())-E.float())))/(T.size(0)*T.size(1)) #/(torch.linalg.norm(E))
    tdvp_error = torch.real(torch.linalg.norm((torch.mv(Ore.float(),δ.float()))-E.real.float()))/E.size()[0] #/(torch.linalg.norm(E))
    return δ, inversion_error, tdvp_error

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
    Ore, Oim, indices = compute_centered_jacobian(model, samples)
    # Compute the gradients of the NN parameters from that.
    sparse_grads, inversion_error, tdvp_error = compute_gradient_with_curvature(Ore, Oim, E, model, epsilon)
    # Assign the calculated gradients and make one optimization step.
    cost = torch.linalg.norm(sparse_grads).detach().cpu().numpy()
    if cost >= 1000:
        print("Gradient = "+str(cost)+" --> Cut gradient.")
        sparse_grads *= 1e-8/cost
    apply_grads(model,sparse_grads,indices)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return cost, inversion_error, tdvp_error
