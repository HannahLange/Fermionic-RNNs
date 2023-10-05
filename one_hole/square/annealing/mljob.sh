#!/bin/bash
#SBATCH --job-name=annealing
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
python3 run.py
