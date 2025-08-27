#!/bin/bash
#SBATCH -J vae_train             # Job name
#SBATCH -p edu-medium            # Partition (queue)
#SBATCH -N 1                     # 1 node
#SBATCH -n 1                     # 1 task
#SBATCH -c 4                     # 4 CPU cores
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --mem=16G                # 16 GB RAM
#SBATCH -t 02:00:00              # Time limit (2 hours)
#SBATCH -o logs/%x_%j.out        # Stdout/stderr log file

mkdir -p logs

# Load CUDA only if you rely on system CUDA
# module load CUDA/12.1.1

# ✅ Activate conda (adjust if needed)
source ~/.bashrc
conda activate deepl

# Run your startup script
bash startup.sh