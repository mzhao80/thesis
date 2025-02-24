#!/bin/bash
#SBATCH -J pipeline
#SBATCH -o logs/pipeline_%j.out
#SBATCH -e logs/pipeline_%j.err
#SBATCH -p gpu_requeue
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --account=murphy_lab

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd topic_taxonomy_expansion

# Run the GPU script
python topic_generation.py
python clustering.py
python topic_generation_2.py
python clustering_2.py
python viz_tree.py
python merge_taxonomy.py

cd ../vast
python single_run_subtopic.py
python ideal_points.py --method umap --dimensions 1 --subtopic-level subtopic1
python ideal_points.py --method umap --dimensions 1 --subtopic-level subtopic2
python ideal_points.py --method umap --dimensions 2 --subtopic-level subtopic1
python ideal_points.py --method umap --dimensions 2 --subtopic-level subtopic2
python ideal_points.py --method pca --dimensions 1 --subtopic-level subtopic1
python ideal_points.py --method pca --dimensions 1 --subtopic-level subtopic2
python ideal_points.py --method pca --dimensions 2 --subtopic-level subtopic1
python ideal_points.py --method pca --dimensions 2 --subtopic-level subtopic2