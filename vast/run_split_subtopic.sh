#!/bin/bash

# StrideStance: Run Stance Detection with Document Splitting
#
# This script serves as the primary entry point for running the StrideStance system
# for stance detection with document splitting. It configures and executes the 
# split_run_subtopic.py script with appropriate parameters.
#
# The script handles:
# 1. Setting paths for input/output files and checkpoints
# 2. Configuring model parameters
# 3. Setting document splitting parameters
# 4. Configuring aggregation methods for both chunk and document levels
#
# Usage:
#   ./run_split_subtopic.sh
#
# The parameters below can be modified to customize the StrideStance system
# for different datasets and scenarios.

# Data paths
INPUT_CSV="/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv"
SPLIT_CSV="/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data_split.csv"
OUTPUT_CSV="/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/stance_predictions.csv"

# Create output directories if they don't exist
mkdir -p split_data
mkdir -p results

# Model parameters
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
CHECKPOINT_PATH="ckp_new_new/finetuned_model.pt"
BATCH_SIZE=8

# Document splitting parameters
MAX_LENGTH=384
OVERLAP=50

# Aggregation methods
CHUNK_AGG_METHOD="max_pooling"      # Options: max_pooling, simple_averaging, weighted_averaging
DOCUMENT_AGG_METHOD="max_pooling"   # Options: max_pooling, simple_averaging, weighted_averaging

echo "Running stance prediction with document splitting..."
python split_run_subtopic.py \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE \
  --checkpoint_path $CHECKPOINT_PATH \
  --process_original \
  --max_length $MAX_LENGTH \
  --overlap $OVERLAP \
  --chunk_aggregation_method $CHUNK_AGG_METHOD \
  --document_aggregation_method $DOCUMENT_AGG_METHOD \
  --input_csv $INPUT_CSV \
  --split_csv $SPLIT_CSV \
  --output_csv $OUTPUT_CSV
