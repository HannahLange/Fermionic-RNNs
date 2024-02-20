
import os

Nx = 4
Ny = 4

Jp = 0
Jz = 1
t  = 3
bounds_x = 0
bounds_y = 0
if bounds_x == bounds_y:
    bounds = {1:"open",0:"periodic"}[bounds_x]
else:
    bounds = {1:"open",0:"periodic"}[bounds_x]+"_"+{1:"open",0:"periodic"}[bounds_y]
load_data = 1
antisym = 0

for hiddendim in [70]: #10,20,30,50,70]:
    for density in [1-1/(Nx*Ny)]: #,1-4/(Nx*Ny),1-6/(Nx*Ny), 1-8/(Nx*Ny)]: #, 1-1/(Nx*Ny),1-2/(Nx*Ny), 1-3/(Nx*Ny)]:
        fol = str(Nx)+"x"+str(Ny)+"_qubits/"+bounds+"/Jp="+str(float(Jp))+"Jz="+str(float(Jz))+"t="+str(float(t))+"den="+"{:.2f}".format(density)+"/"
        print(fol)
        if not os.path.exists(fol):
            os.mkdir(fol)
        with open("mljob.sh", "r") as f:
            data = f.read()
        data = data.replace(".py", ".py -Nx "+str(Nx)+" -Ny "+str(Ny)+" -den "+str(density)+" -t "+str(t)+" -Jp "+str(Jp)+" -Jz "+str(Jz)+" -boundsx "+str(bounds_x)+" -boundsy "+str(bounds_y)+" -load "+str(load_data)+" -antisym "+str(antisym)+" -hd "+str(hiddendim)+" -sym 0")
        data = data.replace("ml_test", fol.split("/")[-1])
        with open(fol+"/mljob.sh", "w") as f:
            data = f.write(data)
        os.system("sbatch "+fol+"/mljob.sh")
