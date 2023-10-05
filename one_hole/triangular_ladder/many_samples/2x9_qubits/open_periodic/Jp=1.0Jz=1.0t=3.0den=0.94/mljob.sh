#!/bin/bash
#
#SBATCH --job-name=triangular
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
python3 run.py -Nx 2 -Ny 9 -den 0.9444444444444444 -t 3 -Jp 1 -Jz 1 -boundsx 1 -boundsy 0 -load 0 -antisym 0 -hd 100 -sym 0
python3 tests.py -Nx 2 -Ny 9 -den 0.9444444444444444 -t 3 -Jp 1 -Jz 1 -boundsx 1 -boundsy 0 -load 0 -antisym 0 -hd 100 -sym 0
