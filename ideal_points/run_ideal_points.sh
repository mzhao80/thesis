#!/bin/bash

# StrideStance: Ideal Points Pipeline
#
# This script runs the complete pipeline for generating stance-based ideal points
# from speech stance detection outputs. The pipeline consists of three main stages:
#
# 1. PCA Analysis:
#    - Generates 2D PCA embeddings from stance data
#    - Creates visualization of legislators in ideological space
#    - Identifies key dimensions of variation in stance patterns
#
# 2. IRT Model:
#    - Applies a Bayesian Item Response Theory model to stance data
#    - Generates 1D ideal point estimates with uncertainty intervals
#    - Extracts discrimination parameters for topics
#
# 3. Comparative Analysis:
#    - Compares stance-based ideal points with DW-NOMINATE scores
#    - Calculates correlation statistics
#    - Identifies legislators with greatest discrepancies
#
# The script applies filtering to focus on legislators who have spoken on a substantial
# number of topics and topics that have been addressed by a sufficient number of speakers.
# These thresholds can be adjusted in the parameters below.
#
# Usage:
#   ./run_ideal_points.sh
#
# The outputs include CSV files with ideal point estimates and visualizations in PNG format.

# Run the ideal points embedding generation with various options

# Aggregation method options
# Options: simple_averaging, weighted_averaging, max_pooling
AGGREGATION_METHOD="simple_averaging"

# Dimensionality reduction method options
# Options: pca, umap, svd
METHOD="pca"

# Dimensionality options
# Options: 1, 2, 3
DIMENSIONS=2

# Subtopic level options
# Options: subtopic1, subtopic2
SUBTOPIC_LEVEL="subtopic1"

echo "==== Running ideal points embedding generation ===="
echo "Aggregation method: $AGGREGATION_METHOD"
echo "Dimensionality reduction: $METHOD"
echo "Dimensions: $DIMENSIONS"
echo "Subtopic level: $SUBTOPIC_LEVEL"

python ideal_points.py \
    --method $METHOD \
    --dimensions $DIMENSIONS \
    --subtopic-level $SUBTOPIC_LEVEL \
    --aggregation-method $AGGREGATION_METHOD

python irt.py

echo "==== Completed ideal points embedding generation ===="
