#!/bin/bash
#SBATCH -J wandb_agent
#SBATCH -o logs/agent_%j.out
#SBATCH -e logs/agent_%j.err
#SBATCH -p gpu_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --account=murphy_lab

# Change to your project directory
cd ~/Downloads/thesis/topic_taxonomy_expansion

# Activate your virtual environment and load CUDA
source ../../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sweep ID from wandb (update with your actual sweep ID if needed)
SWEEP_ID="michaelzhao-harvard-university/thesis-topic_taxonomy_expansion/qd53gsh6"

# Launch two wandb agents concurrently. Each agent will pick different hyperparameter configurations.
wandb agent ${SWEEP_ID} &
wandb agent ${SWEEP_ID} &

# Wait for both agents to finish (they will typically run until you cancel the job)
wait
