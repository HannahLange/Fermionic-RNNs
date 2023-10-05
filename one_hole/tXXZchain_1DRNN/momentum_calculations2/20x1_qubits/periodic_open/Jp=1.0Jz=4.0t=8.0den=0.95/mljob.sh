#!/bin/bash
#SBATCH --job-name=momentum
#SBATCH --comment=""
#SBATCH --mem=64GB
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --partition=a40,rtx6000,t4v2
source ~/ML_Environment/bin/activate
python3 run.py -Nx 20 -Ny 1 -kx 1.0 -ky 0 -den 0.95 -t 8 -Jp 1 -Jz 4 -boundsx 0 -boundsy 1 -load 1 -antisym 0 -hd 100 -sym 0
