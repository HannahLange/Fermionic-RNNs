#!/bin/bash
#SBATCH --job-name=
#SBATCH --comment=""
#SBATCH --mem=64GB
#SBATCH --qos=normal
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --partition=a40,rtx6000,t4v2
source ~/ML_Environment/bin/activate
python3 run.py -Nx 4 -Ny 4 -kx 0.0 -ky 0.0 -den 0.9375 -t 3 -Jp 1 -Jz 1 -boundsx 1 -boundsy 1 -load 0 -antisym 0 -hd 70 -sym 0
