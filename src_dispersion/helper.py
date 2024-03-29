import torch
import numpy as np
import random
import observables as o
from localenergy import get_Eloc, tJ2D_MatrixElements
import os


def momentum_cost_fct(samples, model, device, H, params, bounds_x, bounds_y, kx, ky, lamb):
    cost = 0
    Eloc, log_probs, phases = get_Eloc(H,params,samples, model, bounds_x, bounds_y, None, False)
    log_psi = (0.5*log_probs+1j*phases)
    eloc_sum = (Eloc).mean(axis=0)

    px, py, expect_x, expect_y = o.calculate_momentum(samples, log_probs, phases, model)
    px = px/(torch.pi)
    py = py/(torch.pi)
    e_loc_corr = Eloc - eloc_sum
    cost += 2 * torch.real((torch.conj(log_psi) * (e_loc_corr.detach().to(torch.complex128)))).mean(axis=0)
    cost += lamb * ((torch.abs(torch.real(px))-kx)**2+(torch.abs(torch.real(py))-ky)**2)/((kx**2+ky**2+0.1)*10)
    if ky!=0: #for some momenta its better to explicitely enforce Im(k)=0
        cost += 0.05*lamb* ((torch.abs(torch.imag(px))-0)**2+(torch.abs(torch.imag(py))-0)**2)
    return Eloc, cost, log_probs, phases, px.mean(), py.mean()


def mse_loss(input, target):
    return (input-target)**2

def save(model, boundaries, folder, fol_extension, n_samples, device):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder+"model_params"+fol_extension+".pt")
    # calculate the nearest neighbor spin correlators
    samples = model.sample(n_samples)
    log_probs, phases = model.log_probabilities(samples)


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
    parser.add_argument("-kx",  "--kx", type=float, default = np.pi/2 , help="kx")
    parser.add_argument("-ky",  "--ky", type=float, default = np.pi/2 , help="ky")
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
    kx         = args.kx
    ky         = args.ky
    antisym    = {1: True, 0: False}[args.antisym]
    sz_tot  = (Nx*Ny*density/2)-int(Nx*Ny*density/2)
    if t!=0 and Jz == 0 and Jp == 0: sz_tot  = (Nx*Ny*density/2)
    sz_tot = np.round(sz_tot,1)
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
    return {"U": U, "t": t, "Jp": Jp, "Jz": Jz}, density, Nx, Ny, kx, ky, b_dict[args.boundsx], b_dict[args.boundsy], {1: True, 0: False}[args.load_model], sz_tot, sym, antisym, hiddendim


def save_params(params, folder, fol_ext):
    with open(folder+"ml_params"+fol_ext+".csv", "w") as f:
        for key in params.keys():
            f.write(key+": "+str(params[key])+"\n")
