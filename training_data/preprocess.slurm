#!/bin/bash
#SBATCH -J preprocess
#SBATCH -o logs/preprocess_%j.out
#SBATCH -e logs/preprocess_%j.err
#SBATCH -p serial_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --account=murphy_lab

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
cd training_data
python preprocess.py
