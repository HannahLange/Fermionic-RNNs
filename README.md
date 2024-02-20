# Fermionic RNNs and NQS quasiparticle dispersions
<div align="center">
    <img width="444" alt="RNN_architecture" src="https://github.com/HannahLange/Fermionic-RNNs/assets/82364625/00e67999-cf2b-4da4-b307-4eec2a4eda65">
</div>



RNN implementation to simulate the bosonic and fermionic 2D $t-J$ model, or its general form, the $t$-XXZ model, using pytorch, see ([our paper](https://arxiv.org/abs/2310.08578)). The code for the ground state search of bosonic and fermionic systems and an run.py (run_sr.py) file to run it (with minSR) can be found [here](https://github.com/HannahLange/Fermionic-RNNs/tree/main/src). All data shown in the paper is provided [here](https://github.com/HannahLange/Fermionic-RNNs/tree/main/data).

In order to run the code, run.py (or run_sr.py and stoch_reconfig.py), helper.py, observables.py,localenergy.py and model.py have to be in the same directory. Furthermore, you need to create a folder for the results, e.g. `mkdir 4x4_qubits/open` for a system with 4x4 sites and open boundaries. You can run the code by calling e.g.

`run.py -Nx 4 -Ny 4 -den 1 -t 3 -Jp 1 -Jz 1 -boundsx 0 -boundsy 0 -load 0 -antisym 0 -hd 50 -sym 0`

for a bosonic $N_x\times N_y=4\times 4$ square lattice system with open boundaries (boundsx=boundsy=0), $t=3$, $J_{\pm}=J_{z}=1$ and hidden dimension $h_d=50$. 

Furthermore, the one-hole [dispersions](https://github.com/HannahLange/Fermionic-RNNs/tree/main/src_dispersion) can be calculated by enforcing a target momentum $k_\mathrm{target}$ by adding this constrain to the cost function:

<div align="center">
    <img width="479" alt="Momentum_git" src="https://github.com/HannahLange/Fermionic-RNNs/assets/82364625/f899bb40-fa28-4569-a13d-583eb30cafa8">
</div>
