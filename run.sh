#!/bin/bash
#SBATCH --job-name=CUB
#SBATCH --output=./R-%x.%j.out

#SBATCH --time=10-00:00:00
#SBATCH --mem=50G  # 120G

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # 16, 20
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=scgn[03-08]

python -m main.py




