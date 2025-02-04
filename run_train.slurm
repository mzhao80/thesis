#!/bin/bash
#SBATCH -J train
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -p gpu_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the GPU script
python topic_taxonomy_expansion/train.py --config topic_taxonomy_expansion/config.json
# python topic_taxonomy_expansion/train.py --config topic_taxonomy_expansion/config_direct.json
