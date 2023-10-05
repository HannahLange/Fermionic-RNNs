import torch
import numpy as np
import random
import observables as o
from localenergy import get_Eloc, tJ2D_MatrixElements, FH2D_MatrixElements
import os
import pandas as pd
import itertools


def cost_fct(samples, model, device, H, params, bounds_x, bounds_y, mu_sym, symmetry, T, antisym):
    cost = 0
    if symmetry != None:
        Eloc, log_probs, phases, sym_samples, sym_log_probs, sym_phases = get_Eloc(H,params,samples, model, bounds_x, bounds_y, symmetry, antisym)
        sym_log_psi = (0.5*sym_log_probs+1j*sym_phases)
    else:
        Eloc, log_probs, phases = get_Eloc(H,params,samples, model, bounds_x, bounds_y, symmetry, antisym)
    log_psi = (0.5*log_probs+1j*phases)
    eloc_sum = (Eloc).mean(axis=0)
    e_loc_corr = Eloc - eloc_sum
    if T != None and T!= 0:
        cost += 4*T*(torch.mean(torch.real(torch.conj(log_psi)*log_probs.detach().to(torch.complex128))))-torch.mean(torch.real(torch.conj(log_psi)*torch.mean(log_probs.detach().to(torch.complex128))) )
    if symmetry != None and mu_sym != 0:
        p_diff = (torch.exp(log_probs)-torch.exp(sym_log_probs))/(0.5*(torch.exp(log_probs)+torch.exp(sym_log_probs)))
        #cost += mu_sz * 400 * torch.real((p_diff).detach() * (torch.real((torch.conj(log_psi)-(torch.conj(sym_log_psi)))))).mean(axis=0)
        #e_loc_corr += (mu_sym *(p_diff.detach())**2)
        p_diff = torch.abs(p_diff.detach())
        #print(((p_diff)).mean())
        #print((mu_sym  * (p_diff)**2).mean())
    else:
        p_diff = None
    cost += 2 * torch.real((torch.conj(log_psi) * (e_loc_corr.detach().to(torch.complex128)))).mean(axis=0)
    return Eloc, cost, log_probs, phases, p_diff

def save(model, boundaries, folder, fol_extension, n_samples, device):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder+"model_params"+fol_extension+".pt")
    # calculate the nearest neighbor spin correlators
    samples = model.sample(n_samples)
    log_probs, phases = model.log_probabilities(samples)
    #szsz = (o.get_szsz(samples, log_probs, boundaries, model, device))
    #np.save(folder+"szsz"+fol_extension+".npy", np.array(szsz))
    #sxsx = (o.get_sxsx(samples, log_probs, phases, boundaries, model, device))
    #np.save(folder+"sxsx"+fol_extension+".npy", np.array(sxsx))
    #sysy = (o.get_sysy(samples, log_probs, phases, boundaries, model, device))
    #np.save(folder+"sysy"+fol_extension+".npy", np.array(sysy))


def initialize_torch():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(0)
    return device


def parse_input(parser, H):
    # Define model parameters
    parser.add_argument("-Jp", "--Jp", type=float , default = 1.0 , help="Jp")
    parser.add_argument("-Jz", "--Jz", type=float , default = 1.0 , help="Jz")
    parser.add_argument("-U", "--U", type=float , default = 1.0 , help="U")
    parser.add_argument("-t", "--t", type=float , default = 1.0 , help="t")
    parser.add_argument("-den", "--density", type=float , default = 1.0 , help="density of particles")
    parser.add_argument("-Nx",  "--Nx", type=int, default = 4 , help="length in x dir")
    parser.add_argument("-Ny",  "--Ny", type=int, default = 4 , help="length in y dir")
    parser.add_argument("-bounds","--bounds", type=int , default = 1 , help="type of boundaries (open:1, periodic: 0)")
    parser.add_argument("-boundsx","--boundsx", type=int , default = 1 , help="type of boundaries in x-dir. (open:1, periodic: 0)")
    parser.add_argument("-boundsy","--boundsy", type=int , default = 1 , help="type of boundaries in y-dir. (open:1, periodic: 0)")
    parser.add_argument("-load", "--load_model", type=int , default = 1 , help="load previous model if 1")
    parser.add_argument("-sym", "--sym", type=float , default = 1 , help="enforces symmetries if 1")
    parser.add_argument("-antisym", "--antisym", type=float , default = 1 , help="antisymmetry if  1")
    parser.add_argument("-hd", "--hd", type=int , default = 50 , help="hidden dimension")
    args = parser.parse_args()
    print(args)
    hiddendim = args.hd
    Jp         = args.Jp
    Jz         = args.Jz
    U          = args.U
    t          = args.t
    density    = args.density
    Nx         = args.Nx
    Ny         = args.Ny
    antisym    = {1: True, 0: False}[args.antisym]
    sz_tot  = (Nx*Ny*density/2)-int(Nx*Ny*density/2)
    if t!=0 and Jz == 0 and Jp == 0: sz_tot  = (Nx*Ny*density/2)
    if density == 1 and H=="tJ": t = 0
    if args.sym == 1:
        if Nx == Ny:
            sym     = "C4"
        else:
            sym = "C2"
        print("Enforce symmetries: C4 and Sz_tot = "+str(sz_tot))
    else:
        sym     = None
    b_dict = {1: "open", 0: "periodic"}
    if args.bounds == 0:
        args.boundx = 0
        args.boundy = 0
    return {"U": U, "t": t, "Jp": Jp, "Jz": Jz}, density, Nx, Ny, b_dict[args.boundsx], b_dict[args.boundsy], {1: True, 0: False}[args.load_model], sz_tot, sym, antisym, hiddendim


def save_params(params, folder, fol_ext):
    with open(folder+"ml_params"+fol_ext+".csv", "w") as f:
        for key in params.keys():
            f.write(key+": "+str(params[key])+"\n")


def get_ampl(samples, phases):
    samples_ = samples.clone()
    samples_ = torch.reshape(samples_, (samples.size(0), samples.size(1)*samples.size(2)))
    indices = torch.stack([torch.arange(0,samples_.size(1), device=samples_.device) for idx in range(samples_.size(0))], axis=0)
    holes = torch.where(samples_ == 0, indices, -10)
    try:
        holes = torch.reshape(holes[holes!=-10], (holes.size(0),-1))
    except RuntimeError:
        holes = torch.stack([torch.empty(0) for idx in range(samples.size(0))], axis=0)
    particles = torch.where(samples_!=0, samples_, -10)
    particles = torch.reshape(particles[particles!=-10], (particles.size(0),-1))
    indices_p = torch.stack([torch.arange(0,particles.size(1), device=samples_.device) for idx in range(particles.size(0))], axis=0)
    up_particles = torch.where(particles == 1, indices_p, -10) 
    up_particles = torch.reshape(up_particles[up_particles!=-10], (up_particles.size(0),-1))
    idx = index_config(holes, up_particles, samples_.size(1)-holes.size(1), up_particles.size(1),samples_.size(1),holes.size(1),samples.device)
    phases = phases[idx]

    #print(samples, holes, up_particles, phases)
    return torch.reshape(phases, (phases.size(0),))

def index_config(hole_pos, up_pos, Lspin, Nup, L, Nh,device):
    # takes a Fock space configuration (given by position of holes hole_pos and
    # position of ups up_pos) and outputs the corresponding index
    # (1<=index<=hilbert space dimension) in our Hilbert space 
    # hole_pos, up_pos are arrays with integers (sites go from 0 to L-1 or 0 to
    # Lspin-1)

    spin_L_N = torch.sum(2 ** torch.tensor(list(itertools.combinations(range(Lspin), Nup)), device=device), dim=1)
    spincount_L_N = len(spin_L_N)

    # Generate hole configurations
    hole_L_N = torch.sum(2 ** torch.tensor(list(itertools.combinations(range(L), Nh)), device=device), dim=1)
    holecount_L_N = len(hole_L_N)

    # Get spin index
    a = torch.sum(2 ** up_pos, axis=-1)
    j = torch.stack([torch.where(spin_L_N == a[idx])[0] for idx in range(a.size(0))])

    # Get hole index
    b = torch.sum(2 ** hole_pos, axis=-1)
    h = torch.stack([torch.where(hole_L_N == b[idx])[0] for idx in range(b.size(0))])
    # Combine spins and holes
    idx = (h) * spincount_L_N + j
    #print(hole_pos, up_pos, idx)
    return idx.to(torch.long)
