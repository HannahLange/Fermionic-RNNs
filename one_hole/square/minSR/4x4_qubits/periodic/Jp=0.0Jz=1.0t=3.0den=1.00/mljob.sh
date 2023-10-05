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
#SBATCH --partition=a40
source ~/ML_Environment/bin/activate
python3 run_sr.py -Nx 4 -Ny 4 -den 1 -t 3 -Jp 0 -Jz 1 -boundsx 0 -boundsy 0 -load 0 -antisym 0 -hd 70 -sym 0
