import os
import numpy as np

Nx = 4
Ny = 4

Jp = 1
Jz = 1
t  = 3

density = 1-1/(Nx*Ny)

bounds_x = 0
bounds_y = 0
if bounds_x == bounds_y:
    bounds = {1:"open",0:"periodic"}[bounds_x]
else:
    bounds = {1:"open",0:"periodic"}[bounds_x]+"_"+{1:"open",0:"periodic"}[bounds_y]
load_data = 1
antisym = 0

hiddendim = 100
#define paths
for kx in np.arange(0,0.5+2/Nx,2/Nx):
    for ky in np.arange(0,0.5+2/Nx,2/Ny):
        if not ((kx==1 or ky==1) and kx==ky):
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
