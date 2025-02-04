#!/bin/bash
#SBATCH --partition=serial_requeue        # Use the requeue partition
#SBATCH --cpus-per-task=4           # Request 4 CPUs
#SBATCH --mem=10G                   # Request 10 GB of memory
#SBATCH --time=1-00:00:00           # Set the maximum runtime (1 day)
#SBATCH --open-mode=append
#SBATCH --output=logs/fetch_metadata_%j.out       # Standard output log file (with job ID)
#SBATCH --error=logs/fetch_metadata_%j.err        # Standard error log file (with job ID)
#SBATCH --job-name=fetch_metadata # Job name

cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate

python fetch_metadata.py