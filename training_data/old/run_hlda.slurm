#!/bin/bash
#SBATCH -J hlda_keywords
#SBATCH -o logs/hlda_%j.out
#SBATCH -e logs/hlda_%j.err
#SBATCH -p serial_requeue
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=20G

cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate

# Run the hLDA script
python keywords_hlda.py
