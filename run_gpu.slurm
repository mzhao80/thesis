#!/bin/bash
#SBATCH -J gpu_keywords
#SBATCH -o logs/gpu_%j.out
#SBATCH -e logs/gpu_%j.err
#SBATCH -p gpu_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --gres=gpu:2

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01

# Run the GPU script
python keywords_gpu.py
