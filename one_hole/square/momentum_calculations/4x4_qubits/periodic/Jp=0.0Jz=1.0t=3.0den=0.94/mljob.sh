#!/bin/bash
#SBATCH --job-name=momentum
#SBATCH --comment=""
#SBATCH --mem=64GB
#SBATCH --qos=normal
#SBATCH --time=50:00:00
#SBATCH --mail-type=ALL
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --partition=a40,rtx6000,t4v2
source ~/ML_Environment/bin/activate
python3 run.py -Nx 4 -Ny 4 -kx 0.5 -ky 0.5 -den 0.9375 -t 3 -Jp 0 -Jz 1 -boundsx 0 -boundsy 0 -load 1 -antisym 0 -hd 100 -sym 0