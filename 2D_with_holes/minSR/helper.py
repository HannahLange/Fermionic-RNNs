import torch
import numpy as np
import random
import observables as o
from localenergy import get_Eloc, tJ2D_MatrixElements, FH2D_MatrixElements
import os

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
    szsz = (o.get_szsz(samples, log_probs, boundaries, model, device))
    np.save(folder+"szsz"+fol_extension+".npy", np.array(szsz))
    sxsx = (o.get_sxsx(samples, log_probs, phases, boundaries, model, device))
    np.save(folder+"sxsx"+fol_extension+".npy", np.array(sxsx))
    sysy = (o.get_sysy(samples, log_probs, phases, boundaries, model, device))
    np.save(folder+"sysy"+fol_extension+".npy", np.array(sysy))


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
    args = parser.parse_args()
    print(args)

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
    return {"U": U, "t": t, "Jp": Jp, "Jz": Jz}, density, Nx, Ny, b_dict[args.boundsx], b_dict[args.boundsy], {1: True, 0: False}[args.load_model], sz_tot, sym, antisym


def save_params(params, folder, fol_ext):
    with open(folder+"ml_params"+fol_ext+".csv", "w") as f:
        for key in params.keys():
            f.write(key+": "+str(params[key])+"\n")
