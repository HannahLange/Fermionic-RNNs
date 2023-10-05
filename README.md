# Fermionic-RNNs
<img width="600" alt="Convergence_overview_tJz" src="https://github.com/HannahLange/Fermionic-RNNs/assets/82364625/1a4e3f1e-0280-4e79-9dad-270b6cb13d37">


RNN implementation to simulate the bosonic and fermionic t-J model. The data shown in the paper and code for 0 and 1 holes are provided in "one_hole/", for several holes in "more_holes/".

In order to run the Code, run.py (or run_sr.py and stoch_reconfig.py), helper.py, observables.py,localenergy.py and model.py are needed. You can run the code by calling e.g.

`run.py -Nx 4 -Ny 4 -den 1 -t 3 -Jp 1 -Jz 1 -boundsx 0 -boundsy 0 -load 0 -antisym 0 -hd 50 -sym 0`

for a bosonic $N_x\times N_y=4\times 4$ square lattice system with open boundaries (boundsx=boundsy=0), $t=3$, $J_\pm=J_z=1$ and hidden dimension $h_d=50$. 

Furthermore, the one-hole dispersion can be found e.g. in "one_hole/.../momentum_calculations_full_res/" for the respective system.

