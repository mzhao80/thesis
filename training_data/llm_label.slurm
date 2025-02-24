#!/bin/bash
#SBATCH -J llm_label
#SBATCH -o logs/llm_label_%j.out
#SBATCH -e logs/llm_label_%j.err
#SBATCH -p sapphire
#SBATCH -t 00-04:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --account=murphy_lab

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
cd training_data
python llm_label.py
