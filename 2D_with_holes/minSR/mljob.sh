#!/bin/bash
#
#SBATCH --job-name=minSR
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
python3 run_sr.py
python3 tests.py
