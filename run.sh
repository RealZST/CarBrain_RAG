#!/bin/bash
#SBATCH --time=24:00:00
# SBATCH --cpus-per-task=40
# SBATCH --partition=gpu-a100-80g
# SBATCH --partition=gpu-v100-16g
# SBATCH --partition=gpu-p100-16g
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"

# module load cuda
# module load gcc

srun python run.py
