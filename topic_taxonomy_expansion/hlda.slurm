#!/bin/bash
#SBATCH -J hlda
#SBATCH -o logs/hlda_%j.out
#SBATCH -e logs/hlda_%j.err
#SBATCH -p gpu_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --account=murphy_lab

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd topic_taxonomy_expansion

# Run the GPU script
python hlda.py