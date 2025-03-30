# BARTExpan: Topic Taxonomy Expansion for Political Discourse

This repository contains code implementing the BARTExpan methodology and comparative approaches for automatically generating hierarchical taxonomies from legislative document collections. The system creates a three-level taxonomy with policy areas at the top level, followed by two levels of stance-oriented subtopics that are automatically detected and labeled.

## Overview

This project implements three approaches to generate hierarchical topic taxonomies from document collections:

1. **BARTExpan (Standard Approach)**: A novel two-phase process that uses a fine-tuned BART model, sentence embeddings, clustering, and LLM-guided labels to create a three-level taxonomy with stance-oriented topics.
2. **hLDA Approach**: A Hierarchical Latent Dirichlet Allocation model that directly learns the topic hierarchy through a generative process.
3. **BERTopic Approach**: Uses transformer-based embeddings with hierarchical clustering to create an alternative taxonomy through a bottom-up approach.

Each approach generates a structured taxonomy with:
- Policy areas (top-level categories from Congressional Research Service)
- First-level stance-oriented subtopics
- Second-level stance-oriented subtopics
- Document assignments to each node

## Pipeline Workflow

### BARTExpan (Standard Approach)

1. **Topic Generation (topic_generation.py)**:
   - Generates first-level subtopics from documents using a fine-tuned BART model
   - Produces stance-oriented subtopic labels conditional on parent topics
   - Uses sentence embeddings to represent document semantics

2. **Clustering (clustering.py)**:
   - Clusters first-level subtopics using HDBSCAN and UMAP
   - Generates representative stance-oriented labels for clusters using OpenAI API

3. **Second-level Topic Generation (topic_generation_2.py)**:
   - Generates second-level subtopics based on first-level clusters
   - Creates more fine-grained stance-oriented topic distinctions

4. **Second-level Clustering (clustering_2.py)**:
   - Clusters second-level subtopics 
   - Generates stance-oriented cluster labels using OpenAI API

5. **Taxonomy Merging (merge_taxonomy.py)**:
   - Merges taxonomy files with original training data
   - Prepares final taxonomy structure for evaluation

### Alternative Approaches

- **hLDA Approach (hlda.py)**: 
  - Trains a hierarchical topic model to directly learn the topic structure
  - Creates a three-level taxonomy with document assignments
  - Produces topic labels based on word distributions

- **BERTopic Approach (bertopic_trial.py)**:
  - Uses transformer-based embeddings and hierarchical clustering
  - Automatically generates topic labels from term importance
  - Builds the hierarchy in a bottom-up fashion

### Visualization

- **Taxonomy Visualization (viz_tree.py)**:
  - Creates hierarchical visualizations of the generated taxonomies
  - Shows document counts and percentages at each level

## File Descriptions

- **topic_generation.py**: Generates first-level subtopics from documents using a fine-tuned BART model
- **clustering.py**: Clusters first-level subtopics and assigns stance-oriented labels using GPT
- **topic_generation_2.py**: Generates second-level subtopics from first-level clusters
- **clustering_2.py**: Clusters second-level subtopics and assigns stance-oriented labels
- **merge_taxonomy.py**: Merges taxonomy files for final output
- **hlda.py**: Implements Hierarchical Latent Dirichlet Allocation approach as a baseline
- **bertopic_trial.py**: Implements BERTopic-based taxonomy generation as a baseline
- **viz_tree.py**: Creates hierarchical visualizations of taxonomies
- **mean_embedding.py**: Utility functions for working with embeddings and clustering

## Methodological Innovations

BARTExpan introduces several methodological innovations:

1. **Generative models for conditional topic labeling**: Using a fine-tuned BART model to produce context-sensitive, conditional topic labels.
2. **LLM-guided cluster labeling**: Using OpenAI API (GPT-4o-mini) to create interpretable, stance-oriented labels that capture the essence of each cluster.
3. **Stance-oriented topic formulation**: Topics are framed as directional policy positions (e.g., "Expanding Medicare Coverage" rather than "Healthcare"), creating a natural bridge to stance detection.
4. **Improved density-based clustering**: Using UMAP for dimensionality reduction and HDBSCAN for flexible clustering instead of k-means.

## Dependencies

The project depends on several Python libraries:

```
pandas
numpy
torch
transformers
sentence-transformers
hdbscan
umap-learn
tomotopy
bertopic
graphviz
scikit-learn
openai
matplotlib
seaborn
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the OpenAI API key if using the standard approach:
   ```python
   # Create api_keys.py with:
   OPENAI_API_KEY = "your-api-key-here"
   ```

## Usage

### BARTExpan (Standard Approach)

1. Run the first-level topic generation:
   ```
   python topic_generation.py
   ```

2. Run clustering for first-level topics:
   ```
   python clustering.py
   ```

3. Run second-level topic generation:
   ```
   python topic_generation_2.py
   ```

4. Run clustering for second-level topics:
   ```
   python clustering_2.py
   ```

5. Merge the taxonomy:
   ```
   python merge_taxonomy.py
   ```

### Alternative Approaches

For the hLDA approach:
```
python hlda.py
```

For the BERTopic approach:
```
python bertopic_trial.py
```

### Visualization

To visualize all taxonomies:
```
python viz_tree.py
```

## Output Files

The pipeline generates several output files:

- **step_1.csv**: First-level stance-oriented subtopics
- **step_2.csv**: Clustered first-level subtopics
- **step_3.csv**: Second-level stance-oriented subtopics
- **step_4.csv**: Clustered second-level subtopics (final BARTExpan taxonomy)
- **hlda_taxonomy.csv**: Taxonomy generated by hLDA
- **bertopic_taxonomy.csv**: Taxonomy generated by BERTopic
- **taxonomy_tree.pdf**: Visualization of BARTExpan taxonomy
- **taxonomy_tree_hlda.pdf**: Visualization of hLDA taxonomy
- **taxonomy_tree_bertopic.pdf**: Visualization of BERTopic taxonomy

## Batch Processing

A SLURM script (`generate.slurm`) is provided for running the pipeline on compute clusters.

## Citation

If you use this code or methodology in your research, please cite the original thesis work:

```
[Citation details will be added upon publication]
