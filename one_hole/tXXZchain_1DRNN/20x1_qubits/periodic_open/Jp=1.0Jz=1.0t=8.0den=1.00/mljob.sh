#!/bin/bash
#SBATCH --job-name=onehole_square
#SBATCH --comment=""
#SBATCH --mem=64GB
#SBATCH --qos=normal
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --partition=a40,t4v2,rtx6000
source ~/ML_Environment/bin/activate
python3 run.py -Nx 20 -Ny 1 -den 1 -t 8 -Jp 1 -Jz 1 -boundsx 0 -boundsy 1 -load 0 -antisym 0 -hd 300 -sym 0
