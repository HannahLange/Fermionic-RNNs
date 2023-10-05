#!/bin/bash
#
#SBATCH --job-name=momentum
#SBATCH --comment=""
#SBATCH --mem=15GB
#SBATCH --time=40:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hannah.lange@physik.uni-muenchen.de
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100
#SBATCH --partition=inter
source ../../ML_Environment/bin/activate
python3 run.py -Nx 10 -Ny 4 -kx 0.4 -ky 0.5 -den 0.975 -t 3 -Jp 0 -Jz 1 -boundsx 1 -boundsy 0 -load 1 -antisym 0 -hd 300 -sym 0
#python3 tests.py -Nx 10 -Ny 4 -kx 0.4 -ky 0.5 -den 0.975 -t 3 -Jp 0 -Jz 1 -boundsx 1 -boundsy 0 -load 1 -antisym 0 -hd 300 -sym 0
