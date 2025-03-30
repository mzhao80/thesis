# StrideStance: Knowledge-Enhanced Stance Detection for Long Documents

This repository contains the implementation of StrideStance, a stance detection system specifically designed for long documents like legislative speeches. The model builds upon [He et al. (2022)](https://github.com/zihaohe123/wiki-stance-detection)'s knowledge-enhanced approach with additional innovations for handling lengthy text and improved semantic representations.

## Overview

StrideStance addresses three key challenges in stance detection for political text:

1. **Long Document Processing**: Uses stride-based pooling to handle documents exceeding standard transformer token limits
2. **Semantic Representation**: Integrates advanced sentence transformer models for improved text understanding
3. **Knowledge Enhancement**: Incorporates external knowledge for better context and topic understanding

The system is designed to predict stance (favor, against, neutral) on political topics, with particular focus on zero-shot performance for emerging topics.

## System Architecture

![StrideStance Architecture](../figures/stance_detection.pdf)

The architecture consists of several key components:

1. **Document Chunking**: Split long documents into overlapping chunks to fit within model token limits
2. **Knowledge Integration**: Combine document embeddings with external knowledge about the topic
3. **Stride-based Pooling**: Use max pooling across document chunks to capture the strongest stance signals

## Directory Structure

```
vast/
├── ckp/                   # Original model checkpoints
├── ckp_new_new/           # Fine-tuned model checkpoints
├── data/                  # Training and evaluation data
├── results/               # Model prediction outputs
├── split_data/            # Chunked document data
├── split_finetuning_data/ # Data for fine-tuning with split documents
├── zero-shot-stance/      # Original Zihao He implementation
├── *_datasets.py          # Data loading modules
├── *_engine.py            # Model engines for training and inference
├── *_models.py            # Model architecture implementations
├── run_finetuning_split.sh # Fine-tuning script
├── run_split_subtopic.sh  # Inference script for subtopic stance detection
└── split_long_documents.py # Document chunking implementation
```

## Key Files

- **split_run_subtopic.py**: Main script for running inference on subtopics
- **split_engine_subtopic.py**: Core implementation of the stride-based stance detection
- **split_long_documents.py**: Utility for chunking documents at sentence boundaries
- **finetuning_split_engine.py**: Engine for fine-tuning models on split documents
- **he_models.py**: Base model architectures adapted from He et al.'s implementation

## Code Documentation

The following section provides detailed documentation of the key code files in the StrideStance implementation:

### Core Engine Files

#### `split_engine_subtopic.py`
The core engine of the StrideStance system, implementing:
- Document chunk processing and prediction
- Multiple aggregation methods (max pooling, simple averaging, weighted averaging)
- Confidence scoring for predictions
- Document-level and speaker-subtopic level aggregation of predictions
- Integration with the BERTSeqClf model for stance classification

#### `finetuning_split_engine.py`
Implements the fine-tuning process for the StrideStance approach:
- Training with document chunks while maintaining document-level relationships
- Knowledge-enhanced training for improved topic understanding
- Multiple aggregation methods during training
- Early stopping and model checkpoint saving
- Training metrics tracking and validation-based model selection

### Model Architecture

#### `he_models.py`
Defines the core neural network architecture:
- BERT-based sequence classification model with knowledge enhancement
- Adaptation of pre-trained sentence transformer models
- Integration mechanisms for external knowledge
- Stance prediction layers (against, favor, neutral)

### Document Processing

#### `split_long_documents.py`
Utility for preprocessing long documents:
- Document splitting at sentence boundaries
- Overlapping chunk creation for context preservation
- Handling of extremely long sentences
- Metadata preservation during splitting
- Token length estimation and management

### Data Handling

#### `split_datasets_subtopic.py`
Dataset utilities for the StrideStance system:
- Loading and preprocessing split document chunks
- PyTorch DataLoader creation for efficient batch processing
- Integration of document chunks with subtopic information
- Input data preparation for stance detection models
- Document-chunk relationship tracking

### Entry Point Scripts

#### `run_split_subtopic.sh`
Main script for running inference:
- Configures input/output paths
- Sets model parameters and checkpoint paths
- Configures document splitting parameters
- Sets aggregation methods for both chunk and document levels
- Executes the stance detection pipeline

#### `run_finetuning_split.sh`
Script for fine-tuning the model:
- Configures data paths and output directories
- Sets the base model and checkpoint paths
- Configures document splitting parameters
- Sets training hyperparameters
- Configures the chunk aggregation method

#### `split_run_subtopic.py`
Python implementation for the inference pipeline:
- Command-line argument parsing
- Document splitting coordination
- Model initialization and inference execution
- Results aggregation and output

#### `run_finetuning_split.py`
Python implementation for the fine-tuning pipeline:
- Argument parsing for training configuration
- Dataset preparation and splitting
- Model training and validation
- Checkpoint saving and early stopping

## Usage

### 1. Fine-tuning the Model

To fine-tune a model on split documents:

```bash
./run_finetuning_split.sh
```

This script:
- Takes a dataset with labeled stances
- Splits documents to fit within token limits
- Fine-tunes the model with knowledge enhancement
- Saves the fine-tuned model checkpoint

Key parameters in the script:
- `DATA_PATH`: Path to input data CSV
- `MODEL`: Base sentence transformer model
- `CKPT_PATH`: Path to initial checkpoint
- `SAVE_PATH`: Where to save the fine-tuned model
- `CHUNK_AGG_METHOD`: Method for aggregating chunk predictions (max_pooling, simple_averaging, weighted_averaging)

### 2. Running Inference

To perform stance detection on new documents:

```bash
./run_split_subtopic.sh
```

This script:
- Takes input documents (splitting them if needed)
- Runs inference with the fine-tuned model
- Aggregates predictions across document chunks and subtopics
- Saves results to a CSV file

Key parameters:
- `INPUT_CSV`: Path to input document data
- `SPLIT_CSV`: Path to save split documents
- `OUTPUT_CSV`: Path to save stance predictions
- `MODEL_NAME`: Sentence transformer model
- `CHECKPOINT_PATH`: Path to fine-tuned model
- `CHUNK_AGG_METHOD`: Method for aggregating chunk predictions
- `DOCUMENT_AGG_METHOD`: Method for aggregating document predictions

## Methodology

### Document Splitting

Documents are split into overlapping chunks while preserving sentence boundaries:

```python
# From split_long_documents.py
def split_document_at_sentence_boundaries(document, tokenizer, max_length, overlap):
    # Implementation details
```

### Stance Detection

The core prediction pipeline:

1. **Split documents** into overlapping chunks
2. **Process each chunk** with sentence transformer and knowledge enhancement
3. **Aggregate predictions** across chunks using max pooling or averaging
4. **Output final stance** (favor, against, neutral) with confidence scores

### Aggregation Methods

Three methods are available for combining predictions:

1. **Max Pooling**: Take the chunk with highest prediction confidence
2. **Simple Averaging**: Average probabilities across all chunks
3. **Weighted Averaging**: Weight chunks by confidence or relevance

## Reproducing Results

To reproduce the VAST benchmark results:

1. Set up the required dependencies 
2. Fine-tune the model on the VAST dataset using `run_finetuning_split.sh`
3. Evaluate the model using `he_run_vast.py`

## Credits

This implementation builds upon the work of [He et al. (2022)](https://github.com/zihaohe123/wiki-stance-detection) with additional innovations for handling long documents in the legislative domain.

## Citation

If you use this code in your research, please cite:

```
[Citation will be added upon publication]
```
