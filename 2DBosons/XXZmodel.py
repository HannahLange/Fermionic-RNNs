
import numpy as np
import pyten as p
import os

def xy_to_snake(x,y,width):
    """
    Maps [x,y] lattice coordinates to 1D snake
    """
    return x*width + y


def gen_heisenberg(Jp1, Jz1, Jp2, Jz2, Nx, Ny):
    """
    Helper function, setting up the Heisenberg model
    """
    # lattice
    lat = p.mp.lat.u1.genSpinLattice(Nx*Ny, 0.5)
    # hamiltonian
    ham = []
    for y in range(0, Ny):
        for x in range(Nx-1):
            idx1, idx2 = xy_to_snake(x, y, Ny), xy_to_snake(x+1, y, Ny)
            ham.append(Jp1/2 * (lat.get("sp",idx1)*lat.get("sm",idx2) + lat.get("sm",idx1)*lat.get("sp",idx2)) )
            ham.append(Jz1  *  lat.get("sz",idx1)*lat.get("sz",idx2))
    for x in range(0, Nx):
        for y in range(Ny-1):
            idx1, idx2 = xy_to_snake(x, y, Ny), xy_to_snake(x, y+1, Ny)
            ham.append(Jp1/2 * (lat.get("sp",idx1)*lat.get("sm",idx2) + lat.get("sm",idx1)*lat.get("sp",idx2)) )
            ham.append(Jz1  *  lat.get("sz",idx1)*lat.get("sz",idx2))
    for y in range(0, Ny-1):
        for x in range(Nx-1):
            idx1, idx2 = xy_to_snake(x, y, Ny), xy_to_snake(x+1, y+1, Ny)
            ham.append(Jp2/2 * (lat.get("sp",idx1)*lat.get("sm",idx2) + lat.get("sm",idx1)*lat.get("sp",idx2)) )
            ham.append(Jz2  *  lat.get("sz",idx1)*lat.get("sz",idx2))


    ham = p.mp.addLog(ham)
    lat.add("H","H",ham)
    print(lat)
    return lat

def dmrg(state, lattice, stage_desc = ["(m 100 x 10)","(m 500 x 20)", "(m 1000 x 30)"]):
    """
    Helper function, running DMRG
    """
    dmrgconf = p.dmrg.DMRGConfig()
    for stage in stage_desc:
        dmrgconf.stages += [p.dmrg.DMRGStage(stage)]
    pdmrg = p.mp.dmrg.PDMRG(state, [lattice.get("H")], dmrgconf)
    opt_states = []
    for stage in stage_desc:
        print(stage)
        mps = pdmrg.run()
        E = p.mp.expectation(mps, lattice.get("H"))
        opt_states.append(mps)
    return opt_states, E

def get_observables(state, lattice, Nx, Ny):
    sxsx_x = [[] for i in range(Ny)]
    sxsx_y = [[] for i in range(Nx)]
    sysy_x = [[] for i in range(Ny)]
    sysy_y = [[] for i in range(Nx)]
    szsz_x = [[] for i in range(Ny)]
    szsz_y = [[] for i in range(Nx)]
    for y in range(Ny):
        for x in range(Nx-1):
            idx1, idx2 = xy_to_snake(x, y, Ny), xy_to_snake(x+1, y, Ny)
            sxsx_i = p.mp.expectation(state, (lattice.get("sp", idx1)+lattice.get("sm", idx1)) * (lattice.get("sp", idx2)+lattice.get("sm", idx2)))
            sxsx_x[y].append(sxsx_i*0.25)
            sysy_i = p.mp.expectation(state, (lattice.get("sp", idx1)-lattice.get("sm", idx1)) * (lattice.get("sp", idx2)-lattice.get("sm", idx2)))
            sysy_x[y].append(-sysy_i*0.25)
            szsz_i = p.mp.expectation(state, lattice.get("sz", idx1) * lattice.get("sz", idx2))
            szsz_x[y].append(szsz_i)
    for x in range(Nx):
        for y in range(Ny-1):
            idx1, idx2 = xy_to_snake(x, y, Ny), xy_to_snake(x, y+1, Ny)
            sxsx_i = p.mp.expectation(state, (lattice.get("sp", idx1)+lattice.get("sm", idx1)) * (lattice.get("sp", idx2)+lattice.get("sm", idx2)))
            sxsx_y[x].append(sxsx_i*0.25)
            sysy_i = p.mp.expectation(state, (lattice.get("sp", idx1)-lattice.get("sm", idx1)) * (lattice.get("sp", idx2)-lattice.get("sm", idx2)))
            sysy_y[x].append(-sysy_i*0.25)
            szsz_i = p.mp.expectation(state, lattice.get("sz", idx1) * lattice.get("sz", idx2))
            szsz_y[x].append(szsz_i)
    return np.array([sxsx_x, sxsx_y]), np.array([sysy_x, sysy_y]), np.array([szsz_x, sxsx_y])

Jp1 = 1.0
Jz1 = 1.0
Jp2 = 0.0
Jz2 = 0.0

Nx = 8
Ny = 8
size = Nx*Ny

if Jz2 == 0 and Jp2 == 0:
    folder = str(Nx)+"x"+str(Ny)+"_qubits/Jp="+str(np.round(Jp1,1))+"Jz="+str(np.round(Jz1,1))+"/"
else:
    folder = str(Nx)+"x"+str(Ny)+"_qubits/Jp="+str(np.round(Jp1,1))+"Jz="+str(np.round(Jz1,1))+"Jp_nn="+str(np.round(Jp2,1))+"Jz_nn="+str(np.round(Jz2,1))+"/"
if not os.path.exists(folder):
    os.makedirs(folder)
print(folder)

lattice = gen_heisenberg(Jp1, Jz1, Jp2, Jz2, Nx, Ny)

if size % 2 == 0:
    channel = "0.0"
else:
    channel = "0.5"
rnd_mps = p.mp.genSampledState(lattice, channel)
opt_states, E = dmrg(rnd_mps, lattice)
opt_states[-1].save(folder+"opt_state.mps")
state = p.mp.MPS(folder+"opt_state.mps")
sxsx, sysy, szsz = get_observables(state, lattice, Nx, Ny)
np.save(folder+"target_sxsx.npy", sxsx)
np.save(folder+"target_sysy.npy", sysy)
np.save(folder+"target_szsz.npy", szsz)
np.save(folder+"/E.npy", E)
