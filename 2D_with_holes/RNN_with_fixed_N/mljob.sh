#!/bin/bash
#
#SBATCH --job-name=ml_test
#SBATCH --comment=""
#SBATCH --mem-per-cpu=64GB
#SBATCH --qos=normal
#SBATCH --time=160:00:00
#SBATCH --mail-type=ALL
#SBATCH --chdir=
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --partition=a40
source ~/ML_Environment/bin/activate
python3 run.py
