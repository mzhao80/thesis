# Stance-Based Ideal Points

This repository contains the implementation of a comprehensive framework for extracting ideological positions from text data, with a particular application to estimating legislators' ideal points from their speeches in the Congressional Record. The framework addresses limitations in traditional vote-based methods (which rely on binary roll-call votes) and existing text-based approaches (which focus on lexical co-occurrence), by leveraging the rich semantic content of legislative discourse.

## Abstract

This thesis presents a framework for analyzing text data to extract ideological positions, with a particular application to estimating legislators' *ideal points* from their speeches. Traditional vote-based methods such as DW-NOMINATE capture limited information from binary roll-call votes, while existing text-based methods rely solely on the *lexical* co-occurrence of words, leaving a gap in leveraging the rich *semantic* content of legislative discourse. To address this, we introduce the first theoretical formulation for *stance-based ideal points* and develop computational methods for their estimation directly from the Congressional Record using language models and contextual embeddings.

Our contributions include: (1) a comprehensive new dataset linking congressional speeches to the bills they discuss; (2) *BARTExpan*, a novel topic taxonomy generation method that discovers *topics* from speeches and organizes them in a hierarchical taxonomy of increasing specificity; (3) *StrideStance*, a state-of-the-art zero-shot stance detection model that identifies legislators' *stances* on these topics; and (4) an integrated framework that combines all of these methodological developments to estimate new *stance-based ideal points* that reveal patterns not captured in voting data.

Applied to the 117th Congress (2021--2023), we validate our method against existing measures, and show that it captures fiscal and economic issues along the primary dimension as well as contemporary social flashpoints on issues such as abortion and civil rights along secondary dimensions. Furthermore, we use these ideal points to illustrate the strategic behavior of swing-district representatives who moderate their speeches while ultimately voting with the party line, as well as to examine the political positions of non-voting delegates.

## Repository Structure

```
thesis/
├── training_data/             # Congressional speech data and preprocessing
│   ├── preprocess.py          # Speech data preprocessing
│   ├── llm_label.py           # Topic labeling using language models
│   └── README.md              # Documentation for the data pipeline
│
├── topic_taxonomy_expansion/  # BARTExpan topic taxonomy generation
│   ├── topic_generation.py    # First-level topic generation
│   ├── clustering.py          # First-level topic clustering
│   ├── topic_generation_2.py  # Second-level topic generation
│   ├── clustering_2.py        # Second-level topic clustering
│   ├── viz_tree.py            # Topic taxonomy visualization
│   └── README.md              # Documentation for BARTExpan
│
├── vast/                      # StrideStance stance detection
│   ├── split_long_documents.py # Document splitting for long speeches
│   ├── split_engine_subtopic.py # Stride-based stance detection engine
│   ├── he_models.py           # Model architecture definitions
│   ├── run_finetuning_split.sh # Fine-tuning script for stance detection
│   ├── run_split_subtopic.sh  # Inference script for stance detection
│   └── README.md              # Documentation for StrideStance
│
├── ideal_points/              # Stance-based ideal point estimation
│   ├── ideal_points.py        # PCA-based dimensionality reduction
│   ├── irt.py                 # Bayesian IRT implementation
│   ├── compare_idealpoints.py # Comparison with DW-NOMINATE
│   ├── run_ideal_points.sh    # Ideal points pipeline script
│   └── README.md              # Documentation for ideal point estimation
│
├── figures/                   # Generated visualizations and figures
├── run.sh                     # Complete end-to-end pipeline script
└── README.md                  # This file
```

## Key Components

### 1. Data Processing & Topic Labeling

The `training_data` directory contains scripts for preprocessing congressional speeches and using language models to generate topic labels. This component:
- Processes the Congressional Record to extract speeches
- Links speeches to bills they discuss
- Generates topic labels using large language models

### 2. BARTExpan: Topic Taxonomy Generation

The `topic_taxonomy_expansion` directory implements the BARTExpan methodology, which:
- Discovers topics from congressional speeches
- Organizes topics in a hierarchical taxonomy
- Provides increasing levels of topic specificity
- Visualizes the topic hierarchy

### 3. StrideStance: Stance Detection

The `vast` directory contains the StrideStance implementation, which:
- Handles long documents through stride-based pooling
- Integrates external knowledge for improved stance detection
- Identifies legislator stances (favor, against, neutral) on topics
- Fine-tunes language models for the congressional domain

### 4. Stance-Based Ideal Points

The `ideal_points` directory implements ideological scaling from stance data:
- PCA-based dimensionality reduction
- Item Response Theory (IRT) model for ideal point estimation
- Comparative analysis with DW-NOMINATE scores
- Visualization of legislators in ideological space

## Installation & Requirements

The codebase requires Python 3.8+ and several libraries:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch and Transformers (for model implementations)
- Sentence-Transformers (for text embeddings)
- PyMC (for Bayesian IRT)
- NumPy, Pandas, Matplotlib, and Seaborn (for data processing and visualization)
- Scikit-learn (for PCA and other algorithms)
- HDBSCAN and UMAP (for clustering)

## Running the Pipeline

The complete pipeline can be executed using the `run.sh` script:

```bash
# Option 1: Run on SLURM cluster
sbatch run.sh

# Option 2: Run locally
bash run.sh
```

This script executes all components in sequence:
1. Data Processing
2. Topic Taxonomy Generation
3. Stance Detection
4. Ideal Point Estimation

For more granular control, you can run individual components separately using their respective scripts, as documented in each component's README.

## Results and Outputs

The pipeline produces several key outputs:
- A hierarchical topic taxonomy in `topic_taxonomy_expansion/output/`
- Stance detection results in `vast/results/`
- Ideal point estimates in `ideal_points/speaker_subtopic1_idealpoints.csv`
- Comparison with DW-NOMINATE in `ideal_points/idealpoint_comparison_data.csv`
- Visualizations in `figures/`

## Citation

If you use this code or dataset in your research, please cite:

```
[Citation will be added upon publication]
```

## License

[License information]

## Acknowledgments

[Acknowledgments]
