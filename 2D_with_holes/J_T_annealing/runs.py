import os

Nx = 4
Ny = 4

Jp = 1
Jz = 1
t  = 3
bounds_x = 1
bounds_y = 1
if bounds_x == bounds_y:
    bounds = {1:"open",0:"periodic"}[bounds_x]
else:
    bounds = {1:"open",0:"periodic"}[bounds_x]+"_"+{1:"open",0:"periodic"}[bounds_y]
load_data = 0
antisym = 1

for density in [0.75]: #, 1-2/(Nx*Ny), 0.5]: #, 1-2/(Nx*Ny), 0.75, 0.50]: #, 1-2/(Nx*Ny)]: #, 0.50, 1-1/(Nx*Ny),1-2/(Nx*Ny), 1-3/(Nx*Ny)]:
    fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(Jp))+"Jz="+str(float(Jz))+"t="+str(float(t))+"den="+"{:.2f}".format(density)+"/"
    print(fol)
    if not os.path.exists(fol):
        os.mkdir(fol)
    with open("mljob.sh", "r") as f:
        data = f.read()
    data = data.replace(".py", ".py -Nx "+str(Nx)+" -Ny "+str(Ny)+" -den "+str(density)+" -t "+str(t)+" -Jp "+str(Jp)+" -Jz "+str(Jz)+" -boundsx "+str(bounds_x)+" -boundsy "+str(bounds_y)+" -load "+str(load_data)+" -antisym "+str(antisym)+" -sym 1")
    data = data.replace("ml_test", fol.split("/")[-1])
    with open(fol+"/mljob.sh", "w") as f:
        data = f.write(data)
    os.system("sbatch "+fol+"/mljob.sh")
