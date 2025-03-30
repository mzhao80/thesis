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

# Complete Pipeline for Stance-Based Ideal Points Framework
# This script runs all components of the thesis project in sequence:
# 1. Data preprocessing and labeling
# 2. Topic taxonomy generation (BARTExpan)
# 3. Stance detection (StrideStance)
# 4. Ideal point estimation

# Load required modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create logs directory if it doesn't exist
mkdir -p logs

# Step 1: Data Processing
echo "====================== STEP 1: DATA PROCESSING ======================"
cd training_data
echo "Running preprocess.py - Processing congressional speech data"
python preprocess.py
echo "Running llm_label.py - Generating topic labels using LLM"
python llm_label.py
cd ..

# Step 2: Topic Taxonomy Generation (BARTExpan)
echo "====================== STEP 2: TOPIC TAXONOMY GENERATION ======================"
cd topic_taxonomy_expansion
echo "Running topic_generation.py - Generating first-level topics"
python topic_generation.py
echo "Running clustering.py - Clustering first-level topics"
python clustering.py
echo "Running topic_generation_2.py - Generating second-level topics"
python topic_generation_2.py
echo "Running clustering_2.py - Clustering second-level topics"
python clustering_2.py

echo "Running hlda.py - Generating topic taxonomy"
python hlda.py
echo "Running bertopic_trial.py - Generating topic taxonomy"
python bertopic_trial.py
echo "Running merge_taxonomy.py - Merging topic taxonomy"
python merge_taxonomy.py
echo "Running viz_tree.py - Visualizing topic taxonomy"
python viz_tree.py
cd ..

# Step 3: Stance Detection (StrideStance)
echo "====================== STEP 3: STANCE DETECTION ======================"
cd vast
echo "Running fine-tuning for stance detection model"
./run_finetuning_split.sh
echo "Running inference on congressional speeches"
./run_split_subtopic.sh
echo "Evaluating stance detection performance"
python he_run_vast.py
cd ..

# Step 4: Ideal Point Estimation
echo "====================== STEP 4: IDEAL POINT ESTIMATION ======================"
cd ideal_points
echo "Running ideal points pipeline"
./run_ideal_points.sh
cd ..

echo "Pipeline complete! Results can be found in:"
echo "- Topic taxonomy: topic_taxonomy_expansion/output/"
echo "- Stance detection: vast/results/"
echo "- Ideal points: ideal_points/speaker_subtopic1_idealpoints.csv"
echo "- Comparison with DW-NOMINATE: ideal_points/idealpoint_comparison_data.csv"