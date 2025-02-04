#!/bin/bash
#SBATCH -J cpu_keywords
#SBATCH -o logs/cpu_%j.out
#SBATCH -e logs/cpu_%j.err
#SBATCH -p serial_requeue
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem=20G

cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01

# Run the CPU script
python keywords_cpu.py
