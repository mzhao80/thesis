"""
StrideStance Split Engine for Subtopics

This file implements the core functionality of the StrideStance model for stance detection
on split documents organized by subtopics. It handles the entire inference pipeline, including:

1. Loading a pre-trained or fine-tuned model
2. Processing document chunks 
3. Aggregating predictions across chunks using stride-based pooling
4. Aggregating predictions by speaker and subtopic

Key features:
- Support for multiple aggregation methods (max pooling, simple averaging, weighted averaging)
- Confidence scoring for predictions
- Preservation of high-confidence chunks for interpretability
- Document-level and speaker-subtopic level aggregation

This engine is designed specifically for the stance detection task on long political
documents that have been split into chunks, with a focus on evaluating stance toward
specific policy subtopics.

Based on the knowledge-enhanced approach from He et al. (2022) with additional 
innovations for handling long documents and improved semantic representations.
"""

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from he_models import BERTSeqClf
from split_datasets_subtopic import split_subtopic_dataloader

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_prediction_weight(probs, method='entropy'):
    """
    Calculate prediction weight using either max probability or entropy.
    
    Args:
        probs: Array of probabilities [prob_against, prob_favor, prob_neutral]
        method: Either 'max_prob' or 'entropy'
    
    Returns:
        Weight between 0 and 1, where higher means more confident
    """
    if method == 'max_prob':
        # Use the maximum probability as weight
        return np.max(probs)
    
    elif method == 'entropy':
        # Calculate entropy: -sum(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probs * np.log(probs + epsilon))
        
        # Normalize entropy to [0, 1] range
        # Max entropy for 3 classes is log(3)
        max_entropy = np.log(3)
        
        # Invert so higher value means less entropy (more confidence)
        normalized_weight = 1 - (entropy / max_entropy)
        return normalized_weight
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")

def calculate_stance_confidence(probs_array, weights):
    """
    Calculate stance confidence based on simple average and the absolute magnitude of stance score.
    Note: The weights parameter is kept for backward compatibility but is no longer used.
    
    Args:
        probs_array: Array of probability arrays, shape (n_chunks, 3)
        weights: Array of weights, shape (n_chunks,) - no longer used
        
    Returns:
        stance: Predicted stance (0, 1, 2)
        confidence: Confidence score for the prediction
        weighted_probs: Simple average probabilities for each class
    """
    # Calculate simple average of probabilities
    avg_probs = np.mean(probs_array, axis=0)
    
    # Get the stance with highest probability
    stance = np.argmax(avg_probs)
    
    # Calculate confidence as absolute magnitude of stance score (difference between favor and against)
    confidence = abs(avg_probs[1] - avg_probs[0])
    
    return stance, confidence, avg_probs

def calculate_stance_score(probs):
    """
    Calculate stance score as the difference between favor and against probabilities.
    
    Args:
        probs: Array of probabilities [prob_against, prob_favor, prob_neutral]
    
    Returns:
        score: Stance score (-1 to 1, where -1 is strongly against and 1 is strongly favor)
        stance: Predicted stance (0=against, 1=favor, 2=neutral)
        confidence: Confidence in the prediction
    """
    # Calculate stance score as favor - against
    score = probs[1] - probs[0]
    
    # Determine stance based on score and neutral probability
    neutral_threshold = 0.1  # If favor and against are within this threshold, consider neutral
    if abs(score) < neutral_threshold and probs[2] > max(probs[0], probs[1]):
        stance = 2  # neutral
        confidence = probs[2]
    else:
        if score > 0:
            stance = 1  # favor
            confidence = probs[1]
        else:
            stance = 0  # against
            confidence = probs[0]
    
    return score, stance, confidence

def aggregate_by_average(probs_list, docs_list=None):
    """
    Aggregate multiple probability distributions by taking the simple average.
    
    Args:
        probs_list: List of probability arrays, each [prob_against, prob_favor, prob_neutral]
        docs_list: Optional list of document texts corresponding to each probability array
    
    Returns:
        stance: Final predicted stance (0=against, 1=favor, 2=neutral)
        confidence: Confidence in the prediction
        probs: The averaged probability distribution
        doc_idx: Index of the most important document (or None if docs_list is None)
    """
    # Convert to numpy array for simpler operations
    probs_array = np.array(probs_list)
    
    # Calculate the simple average of probabilities
    avg_probs = np.mean(probs_array, axis=0)
    
    # Get stance from the averaged probabilities
    stance = np.argmax(avg_probs)
    
    # Calculate stance confidence as absolute magnitude of stance score
    confidence = abs(avg_probs[1] - avg_probs[0])
    
    # Find the most important document (the one with highest probability in the winning direction)
    doc_idx = None
    if docs_list is not None:
        # Find document with highest probability in the winning direction
        winning_probs = [probs[stance] for probs in probs_list]
        doc_idx = np.argmax(winning_probs)
    
    return stance, confidence, avg_probs, doc_idx

def combine_chunks(probs_list, doc_ids, chunk_texts=None, method='max_pooling'):
    """
    Combine document chunk probabilities to document-level probabilities using the specified method.
    
    Args:
        probs_list: List of probability arrays for chunks, each [prob_against, prob_favor, prob_neutral]
        doc_ids: List of document IDs corresponding to each chunk's probability array
        chunk_texts: Optional list of chunk texts for calculating weights based on length
        method: Method to use for combining chunks ('max_pooling', 'simple_averaging', or 'weighted_averaging')
    
    Returns:
        doc_probs: Dictionary mapping document IDs to their probability arrays
    """
    # Group probabilities and texts by document ID
    doc_chunks = defaultdict(list)
    doc_texts = defaultdict(list)
    
    for i, (doc_id, probs) in enumerate(zip(doc_ids, probs_list)):
        doc_chunks[doc_id].append(probs)
        if chunk_texts is not None:
            doc_texts[doc_id].append(chunk_texts[i])
    
    # Combine chunks using the specified method
    doc_probs = {}
    
    for doc_id, chunks_probs in doc_chunks.items():
        if len(chunks_probs) == 1:
            # Only one chunk for this document
            doc_probs[doc_id] = chunks_probs[0]
        else:
            if method == 'max_pooling':
                # Find the chunk with the highest max probability value
                max_values = [np.max(chunk_prob) for chunk_prob in chunks_probs]
                max_chunk_idx = np.argmax(max_values)
                doc_probs[doc_id] = chunks_probs[max_chunk_idx]
                
            elif method == 'simple_averaging':
                # Simple average across all chunks
                doc_probs[doc_id] = np.mean(chunks_probs, axis=0)
                
            elif method == 'weighted_averaging' and chunk_texts is not None:
                # Weight by chunk length in words
                texts = doc_texts[doc_id]
                weights = [len(text.split()) for text in texts]
                total_weight = sum(weights)
                
                # Handle case where all weights are 0
                if total_weight == 0:
                    doc_probs[doc_id] = np.mean(chunks_probs, axis=0)
                else:
                    # Normalize weights
                    weights = [w / total_weight for w in weights]
                    # Calculate weighted average
                    weighted_prob = np.zeros(chunks_probs[0].shape)
                    for w, prob in zip(weights, chunks_probs):
                        weighted_prob += w * prob
                    doc_probs[doc_id] = weighted_prob
            
            else:
                # Default to simple averaging if method is not recognized
                print(f"{method} not recognized. Using simple averaging.")
                doc_probs[doc_id] = np.mean(chunks_probs, axis=0)
    
    return doc_probs

def aggregate_documents(doc_probs, doc_meta, doc_texts=None, method='simple_averaging'):
    """
    Aggregate document scores by speaker and subtopic using the specified method.
    
    Args:
        doc_probs: Dictionary mapping document IDs to probability arrays
        doc_meta: Dictionary mapping document IDs to metadata
        doc_texts: Optional dictionary mapping document IDs to texts for weighted averaging
        method: Method to use for aggregation ('simple_averaging', 'weighted_averaging', or 'max_pooling')
    
    Returns:
        speaker_subtopic_results: Dictionary mapping (speaker, subtopic) to list of probability arrays
        speaker_subtopic_metadata: Dictionary mapping (speaker, subtopic) to metadata
        speaker_subtopic_docs: Dictionary mapping (speaker, subtopic) to list of documents
        speaker_subtopic_texts: Dictionary mapping (speaker, subtopic) to list of texts
    """
    # Group documents by speaker and subtopic
    speaker_subtopic_probs = defaultdict(list)
    speaker_subtopic_docs = defaultdict(list)
    speaker_subtopic_metadata = {}
    speaker_subtopic_texts = defaultdict(list)
    
    for doc_id, probs in doc_probs.items():
        meta = doc_meta[doc_id]
        key = (meta['speaker'], meta['subtopic_2'])
        
        speaker_subtopic_probs[key].append(probs)
        speaker_subtopic_docs[key].append(meta['document'])
        
        if doc_texts is not None:
            speaker_subtopic_texts[key].append(doc_texts.get(doc_id, ""))
        
        if key not in speaker_subtopic_metadata:
            speaker_subtopic_metadata[key] = {
                'speaker': meta['speaker'],
                'subtopic_1': meta['subtopic_1'],
                'subtopic_2': meta['subtopic_2'],
                'chamber': meta['chamber']
            }
    
    # If method is max_pooling, we need to select the document with highest probability
    if method == 'max_pooling':
        for key, probs_list in speaker_subtopic_probs.items():
            if len(probs_list) > 1:
                # Find document with highest max probability
                max_values = [np.max(doc_prob) for doc_prob in probs_list]
                max_doc_idx = np.argmax(max_values)
                # Keep only the document with highest max probability
                speaker_subtopic_probs[key] = [probs_list[max_doc_idx]]
                speaker_subtopic_docs[key] = [speaker_subtopic_docs[key][max_doc_idx]]
                if doc_texts is not None and key in speaker_subtopic_texts:
                    speaker_subtopic_texts[key] = [speaker_subtopic_texts[key][max_doc_idx]]
    
    return speaker_subtopic_probs, speaker_subtopic_metadata, speaker_subtopic_docs, speaker_subtopic_texts

class SplitSubtopicEngine:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', batch_size=8, device=None, 
                 checkpoint_path=None, chunk_aggregation_method='max_pooling', 
                 document_aggregation_method='max_pooling'):
        """
        Initialize the engine for running inference on split documents.
        
        Args:
            model_name: Name of the pre-trained model to use
            batch_size: Batch size for inference
            device: Device to run inference on ('cuda' or 'cpu')
            checkpoint_path: Path to the model checkpoint
            chunk_aggregation_method: Method to aggregate chunks into document scores
                ('max_pooling', 'simple_averaging', or 'weighted_averaging')
            document_aggregation_method: Method to aggregate documents into subtopic scores
                ('simple_averaging', 'weighted_averaging', or 'max_pooling')
        """
        # Set up device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Save parameters
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_aggregation_method = chunk_aggregation_method
        self.document_aggregation_method = document_aggregation_method
        
        # Initialize model
        print('Initializing model...')
        num_labels = 3  # Stance labels: against (0), favor (1), neutral (2)
        self.model = BERTSeqClf(num_labels=num_labels, model=model_name)
        self.model = nn.DataParallel(self.model)
        
        # Load checkpoint if provided
        if checkpoint_path:
            print(f'Loading checkpoint from {checkpoint_path}')
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        print('Model initialized')
        print(f"Chunk aggregation method: {self.chunk_aggregation_method}")
        print(f"Document aggregation method: {self.document_aggregation_method}")
    
    def predict(self, csv_path):
        """
        Run inference on split documents and aggregate results using the specified methods.
        First combines document chunks, then aggregates by speaker and subtopic_2.
        
        Args:
            csv_path: Path to the CSV file with split documents
            
        Returns:
            DataFrame with aggregated results matching the format of single_engine_subtopic.py
        """
        print(f"Loading data from {csv_path}")
        
        # Load the data
        dataloader = split_subtopic_dataloader(
            csv_path=csv_path,
            batch_size=self.batch_size,
            model=self.model_name
        )
        
        print(f"Running inference on {len(dataloader.dataset)} document chunks")
        
        # Run inference
        all_probs = []
        all_document_ids = []
        all_chunks = []
        all_speakers = []
        all_subtopics_1 = []
        all_subtopics_2 = []
        all_documents = []
        all_chambers = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move inputs to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                # Get model outputs
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Convert logits to probabilities
                logits = outputs.detach().cpu().numpy()
                probs = np.array([softmax(logit) for logit in logits])
                
                # Store results
                all_probs.extend(probs)
                all_document_ids.extend(batch['document_id'])
                all_chunks.extend(batch['chunk'])  # These are now lists, not tensors
                all_speakers.extend(batch['speaker'])
                all_subtopics_1.extend(batch['subtopic_1'])
                all_subtopics_2.extend(batch['subtopic_2'])
                all_documents.extend(batch['document'])
                all_chambers.extend(batch['chamber'])
        
        print("Aggregating results...")
        
        # First, track metadata for each document
        document_metadata = {}
        for doc_id, speaker, subtopic1, subtopic2, document, chamber in zip(
            all_document_ids, all_speakers, all_subtopics_1, all_subtopics_2, all_documents, all_chambers
        ):
            if doc_id not in document_metadata:
                document_metadata[doc_id] = {
                    'speaker': speaker,
                    'subtopic_1': subtopic1,
                    'subtopic_2': subtopic2,
                    'document': document,
                    'chamber': chamber
                }
        
        # Use chunk texts for weighted averaging if needed
        chunk_texts = all_chunks if self.chunk_aggregation_method == 'weighted_averaging' else None
        
        # Combine chunks into document scores
        print(f"Combining chunks using {self.chunk_aggregation_method}...")
        doc_probs = combine_chunks(all_probs, all_document_ids, chunk_texts, method=self.chunk_aggregation_method)
        
        # Create document texts for weighted averaging if needed
        doc_texts = None
        if self.document_aggregation_method == 'weighted_averaging':
            doc_texts = {}
            for doc_id, document in zip(all_document_ids, all_documents):
                if doc_id not in doc_texts:
                    doc_texts[doc_id] = document
        
        # Aggregate documents by speaker and subtopic
        print(f"Aggregating documents using {self.document_aggregation_method}...")
        speaker_subtopic_results, speaker_subtopic_metadata, speaker_subtopic_docs, speaker_subtopic_texts = aggregate_documents(
            doc_probs, document_metadata, doc_texts, method=self.document_aggregation_method
        )
        
        # Calculate stance and confidence for each speaker-subtopic2 pair
        subtopic2_predictions = []
        for key, probs_list in speaker_subtopic_results.items():
            speaker, subtopic2 = key
            metadata = speaker_subtopic_metadata[key]
            docs_list = speaker_subtopic_docs[key]
            
            if self.document_aggregation_method == 'max_pooling':
                # We already selected the best document, so just use its probabilities
                stance = np.argmax(probs_list[0])
                final_probs = probs_list[0]
                doc_idx = 0 if docs_list else None
                confidence = abs(final_probs[1] - final_probs[0])
            else:
                # For averaging methods, combine probabilities across documents
                if self.document_aggregation_method == 'weighted_averaging':
                    # Use weighted average based on text length
                    texts = speaker_subtopic_texts.get(key, [])
                    if texts:
                        weights = [len(text.split()) for text in texts]
                        total_weight = sum(weights)
                        
                        if total_weight == 0:
                            # Fall back to simple average if all weights are 0
                            final_probs = np.mean(probs_list, axis=0)
                        else:
                            # Normalize weights
                            weights = [w / total_weight for w in weights]
                            # Calculate weighted average
                            final_probs = np.zeros(probs_list[0].shape)
                            for w, prob in zip(weights, probs_list):
                                final_probs += w * prob
                    else:
                        # Fall back to simple average if no texts
                        final_probs = np.mean(probs_list, axis=0)
                else:
                    # Simple average
                    final_probs = np.mean(probs_list, axis=0)
                
                stance = np.argmax(final_probs)
                confidence = abs(final_probs[1] - final_probs[0])
                
                # Find the most important document (the one with highest probability in the winning direction)
                doc_idx = None
                if docs_list:
                    # Find document with highest probability in the winning direction
                    winning_probs = [probs[stance] for probs in probs_list]
                    doc_idx = np.argmax(winning_probs)
            
            # Map stance to label
            stance_labels = ['against', 'favor', 'neutral']
            
            # Calculate document word count 
            document_word_count = 0
            if self.document_aggregation_method == 'weighted_averaging' and docs_list:
                texts = speaker_subtopic_texts.get(key, [])
                if texts:
                    document_word_count = sum(len(text.split()) for text in texts)
            
            subtopic2_predictions.append({
                'speaker': speaker,
                'chamber': metadata['chamber'],
                'subtopic_1': metadata['subtopic_1'],
                'subtopic_2': metadata['subtopic_2'],
                'document_count': len(docs_list),
                'document_word_count': document_word_count,
                'stance': stance_labels[stance],
                'stance_confidence': confidence,
                'prob_against': final_probs[0],
                'prob_favor': final_probs[1],
                'prob_neutral': final_probs[2],
                'most_important_doc': docs_list[doc_idx] if doc_idx is not None else ""
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(subtopic2_predictions)
        results_df = results_df.sort_values(['speaker', 'subtopic_1', 'subtopic_2'])
        
        print(f"Generated predictions for {len(results_df)} speaker-subtopic pairs")
        return results_df
