#!/bin/bash
#SBATCH -J wiki_single_detection
#SBATCH -o logs/wiki_single_detection%A_%a.out
#SBATCH -e logs/wiki_single_detection%A_%a.err
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

python wiki_stance_detection.py --model dual --wiki_mode summary
