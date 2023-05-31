import os
import time

import numpy as np
import torch
from torch import Tensor
from torch import nn
from model import Model

"""
def autograd(input, params):
    O = torch.autograd.grad(input, params, torch.ones_like(input), allow_unused=True, retain_graph=True, create_graph=False)
    return O

def compute_centered_jacobian(model,samples):
    #Computes O=d Psi/d Theta. 
    l = tuple(filter(lambda p: p.requires_grad, model.parameters()))
    print(l[0][0])
    parameters = l
    log_probs, phases = model.log_probabilities(samples)
    jac_ampl = [torch.cat([j_.flatten() for j_ in autograd(0.5*log_probs[i], parameters) if j_!=None]) for i in range(log_probs.size(0))]
    print(jac_ampl[0])
    jac_phase = [torch.cat([j_.flatten() for j_ in autograd(phases[i], parameters) if j_!= None]) for i in range(phases.size(0))]
    jac_ampl = torch.stack(jac_ampl) 
    jac_phase = torch.stack(jac_phase) 
    jac_ampl -= jac_ampl.mean(axis=0)
    jac_phase -= jac_phase.mean(axis=0)
    jac = torch.complex(jac_ampl, jac_phase) 
    print("-----------------", jac_ampl)
    print("----------")
    return jac

from functorch import vmap

def autograd_vmap(inputs, params):
    def autograd(input):
        O = torch.autograd.grad(input, params, torch.ones_like(input), allow_unused=True, retain_graph=True, create_graph=False)
        return O
    out = vmap(autograd)(inputs)
    return out

def _compute_centered_jacobian_vmap(model, samples):
    # Computes O=d Psi/d Theta. 
    l = tuple(filter(lambda p: p.requires_grad, model.parameters()))
    parameters = l
    log_probs, phases = model.log_probabilities(samples_batch)
    autograd_vmap(log_probs, parameters[:-2])

    jac_ampl = vmap(lambda lp: torch.cat([j_.flatten() for j_ in autograd_vmap(0.5*lp, parameters) if j_!=None]))(log_probs)
    jac_phase = vmap(lambda ph: torch.cat([j_.flatten() for j_ in autograd_vmap(ph, parameters) if j_!=None]))(phases)
    jac_ampl -= jac_ampl.mean(axis=0)
    jac_phase -= jac_phase.mean(axis=0)
    return jac_ampl, jac_phase

"""




from functorch import jacrev, vmap, make_functional, grad

def compute_centered_jacobian(model,samples,batchsize=2000,make_sparse=True):
    # Computes real and imaginary parts of O=d Psi/d Theta. 
    func, parameters = make_functional(model)
    jac_ampl = []
    jac_phase = []
    for batch in range(int(samples.size(0)/batchsize)):
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
    jac = torch.complex(jac_ampl,jac_phase) 
    if make_sparse: # remove zero values
        non_zero_columns = torch.where(jac.sum(axis=0) != 0)[0]
        jac = jac[:,jac.sum(axis=0)!=0]
        assert jac.size(1) == non_zero_columns.size(0)
    else:
        non_zero_columns = torch.tensor([])
    with torch.no_grad():
        jac -= jac.mean(axis=0)
    return jac / np.sqrt(jac.size(0)), non_zero_columns

def invert(M, E, rtol=1e-12, onDevice=False):
    if onDevice:
        ev, V = torch.linalg.eigh(M)
    else:
        ev, V = np.linalg.eigh(M.cpu().detach().numpy())
        ev = torch.tensor(ev, device=M.device)
        V  = torch.tensor(V, device=M.device)
    Vt = torch.conj(V).t()
    invEv = torch.where(torch.abs(ev / ev[-1]) > rtol, 1. / ev, 0.)
    #rho_p = torch.einsum("ij,j", Vt.cfloat(), E)
    #snr = rho_p/torch.sqrt(torch.sum((rho_p-torch.mean(rho_p))**2)/rho_p.size(0)**2)
    #snr = snr.real
    regularizer = 1. / (1. + (rtol * torch.abs(ev[-1]) / torch.abs(ev))**6) 
    #regularizer *= rho_p / (1. + (2 / snr)**6)
    #print(regularizer)
    Minv = torch.einsum("ij,jk,kl", V, torch.diag(invEv*regularizer),Vt)
    print(torch.abs(torch.matmul(Minv, M)-torch.diag(torch.ones(M.size(0),device=M.device))).sum()/(M.size(0)*M.size(1)))
    MinvE = torch.mv(Minv.cfloat(), E)
    return MinvE

"""@torch.no_grad()
def solve_linear_problem(A: Tensor, b: Tensor, rcond: float) -> Tensor:
    #Solve linear problem `A · x = b` where `A` is approximately positive-definite.
    u, s, v = (t.to(A.device) for t in torch.svd(A.cpu()))
    s_inv = torch.where(
        s > rcond * torch.max(s),
        torch.reciprocal(s),
        torch.scalar_tensor(0, dtype=s.dtype, device=s.device),
    )
    return v.cfloat().mv(s_inv.cfloat().mul_(u.cfloat().t().mv(b)))
"""

def compute_gradient_with_curvature(O, E, model):
    E = -(E-E.mean()) / np.sqrt(E.size(0))
    T = torch.matmul(O, torch.conj(O).t())
    T_ = torch.real(T) #+torch.diag(torch.ones(T.size(0), device=T.device)*1e-4)  #torch.diag(1e-4*torch.diag(T.real)) 
    #TinvE = invert(T_, E)  #solve_linear_problem(T_,E,1e-12) 
    Tinv = torch.linalg.pinv(T_.cpu(),rtol=1e-12).real #torch.linalg.pinv(T+torch.diag(torch.ones(T.size(0),device=model.device)*1e-4),rtol=1e-14) 
    Tinv = Tinv.to(model.device).cfloat()
    print(torch.abs(torch.matmul(Tinv, T)-torch.diag(torch.ones(T.size(0),device=T.device))).sum()/(T.size(0)*T.size(1)))
    TinvE = torch.mv(Tinv, E)
    δ = torch.real(torch.mv(torch.conj(O).t(),TinvE))
    inversion_error = torch.real(torch.linalg.norm((torch.mv(T,TinvE)-E)))/(torch.linalg.norm(E))
    tdvp_error = torch.real(torch.linalg.norm((torch.mv(O.real,δ)) - E.real))/(torch.linalg.norm(E))
    return δ, inversion_error, tdvp_error

def apply_grads(model,sparse_grad,non_zero_columns):
    """ Assigns the calculated gradients to the model parameter gradients. """
    i = 0
    # insert zeros again
    num_params = model.num_params
    if non_zero_columns.size(0)!=0:
        grad = torch.zeros(num_params,device=model.device)
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


def run_sr(model, E, samples, optimizer, scheduler=None):
    """ Runs a minSR step. """
    # Campute the real and imaginary part of the jacobian matrix. 
    O, indices = compute_centered_jacobian(model, samples)
    # Compute the gradients of the NN parameters from that.
    sparse_grads, inversion_error, tdvp_error = compute_gradient_with_curvature(O, E, model)
    print("gradients = ", sparse_grads)
    # Assign the calculated gradients and make one optimization step.
    apply_grads(model,sparse_grads,indices)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return torch.linalg.norm(sparse_grads), inversion_error, tdvp_error
