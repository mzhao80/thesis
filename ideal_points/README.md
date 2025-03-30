# StrideStance: Stance-Based Ideal Points

This directory contains the implementation of the stance-based ideal point estimation framework, which converts stance detection outputs into ideal point estimates for legislators across various topics.

## Overview

The StrideStance ideal points system uses stance detection results to position legislators in an ideological space, allowing for comparisons both with traditional roll-call vote-based measures and among legislators who may never cast formal votes (such as non-voting delegates).

The implementation consists of several key components:
1. Processing stance detection outputs
2. Creating speaker-topic matrices
3. Applying dimensionality reduction (PCA)
4. Implementing a stance-based Item Response Theory (IRT) model
5. Comparing results with DW-NOMINATE scores

## Key Files

### Main Scripts

- **`ideal_points.py`**: Primary script for computing PCA-based ideal points from stance data
- **`irt.py`**: Implements a continuous IRT model for stance-based ideal point estimation
- **`compare_idealpoints.py`**: Compares stance-based ideal points with DW-NOMINATE scores
- **`run_ideal_points.sh`**: Shell script to run the complete ideal points pipeline

### Analysis Scripts

- **`analyze_speaker_count.py`**: Analyzes the distribution of speeches per legislator
- **`analyze_speaker_coverage.py`**: Evaluates topic coverage among legislators
- **`analyze_topic_coverage.py`**: Assesses how well each topic is covered by speakers
- **`compare_speakers.py`**: Compares individual speakers' ideal points across different methods

### Data Files

- **`speaker_stance_results.csv`**: Raw stance detection results
- **`speaker_subtopic1_matrix.csv`**: Matrix of speaker stance scores by subtopic
- **`congress_117_party.csv`**: Party and chamber information for 117th Congress members
- **`idealpoint_comparison_data.csv`**: Combined data with stance-based and DW-NOMINATE scores

## Usage

### Running the Complete Pipeline

To run the complete ideal points pipeline:

```bash
./run_ideal_points.sh
```

This script:
1. Generates 2D PCA embeddings from stance data
2. Creates a subtopic grid visualization
3. Runs the IRT model to generate ideal points
4. Compares results with DW-NOMINATE scores

### Running Individual Components

#### Generate PCA-Based Ideal Points

```bash
python ideal_points.py --data_path speaker_stance_results.csv \
                      --subtopic_level subtopic1 \
                      --method pca \
                      --n_components 2 \
                      --speaker_percentile 0.75 \
                      --topic_percentile 0.75
```

#### Generate IRT-Based Ideal Points

```bash
python irt.py --input_csv speaker_subtopic1_matrix.csv \
             --output_csv speaker_subtopic1_idealpoints.csv \
             --party_csv congress_117_party.csv \
             --plot_png stance_1d_plot.png \
             --speaker_percentile 0.75 \
             --topic_percentile 0.75
```

#### Compare with DW-NOMINATE

```bash
python compare_idealpoints.py --stance_csv speaker_subtopic1_idealpoints.csv \
                             --nominate_csv external/nominate_117.csv \
                             --output_csv idealpoint_comparison_data.csv \
                             --output_plot idealpoint_comparison.png
```

## Methodology

### PCA-Based Ideal Points

The PCA approach treats each topic as a feature and each legislator as an observation, seeking to find orthogonal axes that capture the maximum variance in the stance data. The implementation:

1. Loads raw stance detection results
2. Aggregates multiple speeches into a single stance score per speaker-topic pair
3. Creates a speaker-topic matrix
4. Applies hierarchical imputation for missing values
5. Runs PCA to reduce dimensions
6. Visualizes results with party-based coloring

### IRT-Based Ideal Points

The IRT approach implements a continuous Item Response Theory model that treats topics as "items" and speakers as "subjects" with latent ideal points. The implementation:

1. Reads the speaker-topic matrix
2. Filters topics and speakers based on coverage thresholds
3. Builds a Bayesian model using PyMC with:
   - θ_j (theta): Latent ideal point for each speaker
   - a_i: Discrimination parameter for each topic
   - b_i: Difficulty parameter for each topic
   - σ: Error standard deviation
4. Fits the model using MCMC sampling
5. Extracts posterior means and credible intervals
6. Visualizes results with party-based coloring

## Outputs

The system produces several key outputs:
- 2D PCA embeddings of legislators
- 1D IRT ideal point estimates with uncertainty
- Loadings of topics on principal components
- Discrimination parameters from IRT model
- Comparison statistics with DW-NOMINATE scores

## Citation

```
[Citation will be added upon publication]
```
