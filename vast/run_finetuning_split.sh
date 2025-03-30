#!/bin/bash

# StrideStance: Fine-tuning Script for Document Splitting Model
#
# This script configures and executes the fine-tuning process for the StrideStance
# stance detection system. It fine-tunes a pre-trained sentence transformer model
# on split documents to enable effective stance detection on long documents.
#
# The script handles:
# 1. Setting data paths and output directories
# 2. Configuring the base model and checkpoint paths
# 3. Setting document splitting parameters
# 4. Configuring the training hyperparameters
# 5. Setting the chunk aggregation method
#
# Usage:
#   ./run_finetuning_split.sh
#
# The parameters below can be modified to customize the fine-tuning process
# for different datasets and experimental configurations.

# Data path
DATA_PATH="/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data_2023_sample_stance.csv"
#DATA_PATH="taxonomy_data_2023_labeled.csv"
OUTPUT_DIR="split_finetuning_data"

# Model and checkpoint
MODEL="sentence-transformers/all-mpnet-base-v2"
CKPT_PATH="ckp/best_model.pt"
SAVE_PATH="test/finetuned_model.pt"

# Aggregation method
CHUNK_AGG_METHOD="max_pooling"  # Options: max_pooling, simple_averaging, weighted_averaging

# Run fine-tuning
echo "Running fine-tuning with document splitting..."
python run_finetuning_split.py \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --model $MODEL \
  --ckpt_path $CKPT_PATH \
  --save_path $SAVE_PATH \
  --process_original \
  --max_doc_length 384 \
  --overlap 50 \
  --batch_size 4 \
  --lr 5e-5 \
  --epochs 20 \
  --patience 5 \
  --gpu 0 \
  --chunk_aggregation_method $CHUNK_AGG_METHOD
