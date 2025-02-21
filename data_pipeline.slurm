#!/bin/bash
#SBATCH -J data_pipeline
#SBATCH -o logs/data_pipeline_%j.out
#SBATCH -e logs/data_pipeline_%j.err
#SBATCH -p gpu_requeue
#SBATCH -t 02-00:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --account=murphy_lab

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd training_data
echo "Running preprocess.py"
python preprocess.py
echo "Running llm_label.py"
python llm_label.py

cd ../topic_taxonomy_expansion
echo "Running topic_generation.py"
python topic_generation.py
echo "Running clustering.py"
python clustering.py
echo "Running topic_generation_2.py"
python topic_generation_2.py
echo "Running clustering_2.py"
python clustering_2.py
echo "Running viz_tree.py"
python viz_tree.py
