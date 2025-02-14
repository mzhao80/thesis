#!/bin/bash
#SBATCH -J single_stance_detection
#SBATCH -o logs/single_stance_new_%A_%a.out
#SBATCH -e logs/single_stance_new_%A_%a.err
#SBATCH -p gpu_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --account=murphy_lab

# Change to your project directory and load modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd vast

python single_stance_detection_new.py --model mlp
