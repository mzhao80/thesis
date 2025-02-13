#!/bin/bash
#SBATCH -J gpt_stance_detection
#SBATCH -o logs/gpt_stance_detection_%j.out
#SBATCH -e logs/gpt_stance_detection_%j.err
#SBATCH -p serial_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --account=murphy_lab

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd vast

# Run the GPU script
python gpt_stance_detection.py
