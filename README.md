# Fermionic-RNNs
<div align="center">
    <img width="444" alt="RNN_architecture" src="https://github.com/HannahLange/Fermionic-RNNs/assets/82364625/571fb17e-2b20-4574-a060-dcc090394ae5">
</div>


RNN implementation to simulate the bosonic and fermionic 2D $t-J$ model, or its general form, the $t$-XXZ model, using pytorch. The data shown in the paper (and the respective code to generate it) for 0 and 1 holes are provided [for up to one hole](https://github.com/HannahLange/Fermionic-RNNs/tree/main/one_hole), for [more than one hole]([Fermionic-RNNs/more_holes/](https://github.com/HannahLange/Fermionic-RNNs/tree/main/more_holes)).

In order to run the code, run.py (or run_sr.py and stoch_reconfig.py), helper.py, observables.py,localenergy.py and model.py have to be in the same directory. Furthermore, you need to create a folder for the results, e.g. '/4x4_qubits/open/' for a system with 4x4 sites and open boundaries. You can run the code by calling e.g.

`run.py -Nx 4 -Ny 4 -den 1 -t 3 -Jp 1 -Jz 1 -boundsx 0 -boundsy 0 -load 0 -antisym 0 -hd 50 -sym 0`

for a bosonic $N_x\times N_y=4\times 4$ square lattice system with open boundaries (boundsx=boundsy=0), $t=3$, $J_{\pm}=J_z=1$ and hidden dimension $h_d=50$. 

Furthermore, the one-hole dispersion can be found e.g. [here](https://github.com/HannahLange/Fermionic-RNNs/tree/main/one_hole/square/momentum_calculations_full_res) for the 4x4 system. It is calculated by enforcing a target momentum $k_\mathrm{target}$ by adding this constrain to the cost function:

<div align="center">
    <img width="479" alt="Momentum_git" src="https://github.com/HannahLange/Fermionic-RNNs/assets/82364625/a5b90ba8-5e60-43cc-a66b-b6d8bee572a7">
</div>
