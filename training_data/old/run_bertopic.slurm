#!/bin/bash
#SBATCH -J bertopic
#SBATCH -o logs/bertopic_%j.out
#SBATCH -e logs/bertopic_%j.err
#SBATCH -p gpu_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2

cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01

# Run the BERTopic script
python keywords_bertopic.py
