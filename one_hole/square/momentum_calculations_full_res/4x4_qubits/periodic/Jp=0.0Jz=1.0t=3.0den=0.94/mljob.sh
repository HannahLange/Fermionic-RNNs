#!/bin/bash
#
#SBATCH --job-name=onehole
#SBATCH --comment=""
#SBATCH --mem=15GB
#SBATCH --time=40:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hannah.lange@physik.uni-muenchen.de
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --partition=inter
source ../../ML_Environment/bin/activate
python3 run.py -Nx 4 -Ny 4 -kx 1.0 -ky 1.0 -den 0.9375 -t 3 -Jp 0 -Jz 1 -boundsx 0 -boundsy 0 -load 1 -antisym 0 -hd 100 -sym 0
python3 tests.py -Nx 4 -Ny 4 -kx 1.0 -ky 1.0 -den 0.9375 -t 3 -Jp 0 -Jz 1 -boundsx 0 -boundsy 0 -load 1 -antisym 0 -hd 100 -sym 0
