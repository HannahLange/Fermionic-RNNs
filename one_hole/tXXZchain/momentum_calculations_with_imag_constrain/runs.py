import os
import numpy as np

Nx = 20
Ny = 1

Jp = 1
Jz = 1
t  = 8

density = 1-1/(Nx*Ny)

bounds_x = 0
bounds_y = 1
if bounds_x == bounds_y:
    bounds = {1:"open",0:"periodic"}[bounds_x]
else:
    bounds = {1:"open",0:"periodic"}[bounds_x]+"_"+{1:"open",0:"periodic"}[bounds_y]
load_data = 1
antisym = 0

hiddendim = 70
#define paths
for kx in np.arange(0,1+2/Nx,2/Nx):
    for ky in [0]: #np.arange(0,1+2/Ny,2/Ny):
        if not ((kx==1 or ky==1) and kx==ky):
            kx = np.round(kx, 3)
            ky = np.round(ky, 3)
            print(kx, ky)
            fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(Jp))+"Jz="+str(float(Jz))+"t="+str(float(t))+"den="+"{:.2f}".format(density)+"/"
            print(fol)
            if not os.path.exists(fol):
                os.mkdir(fol)
            with open("mljob.sh", "r") as f:
                data = f.read()
            data = data.replace(".py", ".py -Nx "+str(Nx)+" -Ny "+str(Ny)+" -kx "+str(kx)+" -ky "+str(ky)+" -den "+str(density)+" -t "+str(t)+" -Jp "+str(Jp)+" -Jz "+str(Jz)+" -boundsx "+str(bounds_x)+" -boundsy "+str(bounds_y)+" -load "+str(load_data)+" -antisym "+str(antisym)+" -hd "+str(hiddendim)+" -sym 0")
            data = data.replace("ml_test", fol.split("/")[-1])
            with open(fol+"/mljob.sh", "w") as f:
                data = f.write(data)
            os.system("sbatch "+fol+"/mljob.sh")
