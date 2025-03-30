"""
StrideStance Fine-tuning Engine for Split Documents

This module implements the fine-tuning engine for the StrideStance approach, which
enables fine-tuning sentence transformer models on split documents for stance detection.
The implementation focuses on:

1. Training with document chunks while maintaining document-level relationships
2. Implementing knowledge-enhanced training with external knowledge integration
3. Supporting various aggregation methods for chunk predictions during training
4. Applying early stopping and model checkpoint saving based on validation performance

Key features:
- Support for multiple document chunk aggregation methods
- Integration with external knowledge sources
- Fine-tuning of pre-trained sentence transformer models
- Validation-based model selection
- Tracking of training metrics for performance analysis

This engine is crucial for adapting pre-trained models to the stance detection task
while handling the challenges of long documents through the StrideStance approach.
It serves as the training component of the overall StrideStance system.
"""

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from finetuning_split_datasets import split_finetuning_dataloader
from he_models import BERTSeqClf

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

class SplitFinetuningEngine:
    def __init__(self, args, chunk_aggregation_method='max_pooling'):
        """
        Engine for fine-tuning models on split documents.
        
        Args:
            args: Arguments object with the following attributes:
                - gpu: GPU device index
                - model: Model name
                - lr: Learning rate
                - batch_size: Batch size
                - epochs: Number of epochs
                - patience: Early stopping patience
                - seed: Random seed
                - save_path: Path to save the model
                - n_layers_freeze: Number of layers to freeze in the model
                - l2_reg: L2 regularization weight
                - data_path: Path to the data
                - process_original: Whether to split original documents
                - max_doc_length: Maximum document length when splitting
                - overlap: Overlap between chunks when splitting
                - n_workers: Number of workers for DataLoader
                - ckpt_path: Path to the checkpoint to load
            chunk_aggregation_method: Method to aggregate document chunks. Options:
                - 'max_pooling': Use the chunk with highest max probability
                - 'simple_averaging': Average probabilities across chunks
                - 'weighted_averaging': Average weighted by chunk length
        """
        # Set GPU device
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Create directory to save checkpoints
        os.makedirs('ckp', exist_ok=True)
        
        # Load data
        print('Preparing data...')
        train_loader = split_finetuning_dataloader(
            csv_path=args.train_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model=args.model,
            num_workers=args.n_workers,
            process_original=args.process_original,
            max_doc_length=args.max_doc_length,
            overlap=args.overlap,
            shuffle=True
        )
        
        val_loader = split_finetuning_dataloader(
            csv_path=args.val_path,
            batch_size=args.batch_size*2,
            max_length=args.max_length,
            model=args.model,
            num_workers=args.n_workers,
            process_original=args.process_original,
            max_doc_length=args.max_doc_length,
            overlap=args.overlap,
            shuffle=False
        )
        
        # Initialize model
        print('Initializing model...')
        num_labels = 3  # Stance labels: against (0), favor (1), neutral (2)
        model = BERTSeqClf(num_labels=num_labels, model=args.model, n_layers_freeze=args.n_layers_freeze)
        model = nn.DataParallel(model)
        
        # Load pre-trained checkpoint if provided
        if args.ckpt_path:
            print(f'Loading checkpoint from {args.ckpt_path}...')
            state_dict = torch.load(args.ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print('Checkpoint loaded successfully')
        
        # Move model to device
        model.to(device)
        
        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore index -1 (no label)
        
        # Store attributes
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        
        # Store the aggregation method
        self.chunk_aggregation_method = chunk_aggregation_method
        print(f"Using chunk aggregation method: {self.chunk_aggregation_method}")
    
    def train(self):
        """Train the model."""
        print(f"{'-'*30} Starting Training {'-'*30}")
        
        # Validate training data before starting
        print("Validating datasets...")
        # Check if we have any valid labels in the validation set
        val_has_labels = False
        for batch in self.val_loader:
            labels = batch['labels']
            valid_labels = (labels != -1).sum().item()
            if valid_labels > 0:
                val_has_labels = True
                print(f"Validation set has valid labels: {valid_labels} found in batch")
                break
        
        if not val_has_labels:
            print("WARNING: No valid labels found in validation set! Training may fail or give poor results.")
            print("Consider fixing your data to ensure validation set has proper labels.")
        
        # Initialize best model tracking
        best_epoch = 0
        best_epoch_f1 = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        
        # Training loop
        for epoch in range(self.args.epochs):
            print(f"{'*'*30} Epoch: {epoch+1}/{self.args.epochs} {'*'*30}")
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Evaluate on validation set
            val_metrics = self.evaluate('val')
            f1 = val_metrics['f1_macro']
            
            # Check if this is the best model
            if f1 > best_epoch_f1:
                best_epoch = epoch
                best_epoch_f1 = f1
                best_state_dict = copy.deepcopy(self.model.state_dict())
                
                # Save checkpoint immediately
                checkpoint_path = self.args.save_path
                torch.save(best_state_dict, checkpoint_path)
                print(f"Saved new best model with F1: {f1:.4f}")
            
            # Print epoch summary
            print(f"Epoch: {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val F1 Macro: {f1:.4f}")
            print(f"  Best Epoch: {best_epoch+1} | Best F1: {best_epoch_f1:.4f}")
            
            # Early stopping
            if epoch - best_epoch >= self.args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        print('Loading best model for final evaluation...')
        self.model.load_state_dict(best_state_dict)
        
        return best_epoch_f1
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        total_batches = len(self.train_loader)
        
        # Progress bar
        progress_bar = tqdm(enumerate(self.train_loader), total=total_batches,
                           desc="Training", leave=False)
        
        for i, batch in progress_bar:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask, token_type_ids)
            
            # Calculate loss (only for samples with valid labels)
            loss = self.criterion(logits, labels)
            
            # Check for NaN loss and debug
            if torch.isnan(loss).item():
                print(f"WARNING: NaN loss detected in batch {i+1}")
                # Try to identify the issue
                valid_labels = (labels != -1).sum().item()
                print(f"  Valid labels in batch: {valid_labels}/{labels.size(0)}")
                
                if valid_labels == 0:
                    print("  Skipping backward pass for batch with no valid labels")
                    # Use a dummy loss to avoid NaN propagation
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    print("  Proceeding with caution despite NaN loss")
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (i + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log every 10% of batches
            interval = max(total_batches // 10, 1)
            if (i + 1) % interval == 0 or (i + 1) == total_batches:
                print(f"  Batch: {i+1}/{total_batches} | Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / total_batches
        return avg_epoch_loss
    
    def evaluate(self, phase='val'):
        """
        Evaluate the model on validation or test set.
        
        Args:
            phase: 'val' for validation, 'test' for test
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        # Choose the correct loader
        loader = self.val_loader if phase == 'val' else None
        if loader is None:
            raise ValueError(f"No loader available for phase: {phase}")
        
        # Initialize lists to store predictions and labels
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Inference loop
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {phase}", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels']
                
                # Forward pass
                logits = self.model(input_ids, attention_mask, token_type_ids)
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Add to lists
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Filter out samples with no labels (-1)
        valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
        if not valid_indices:
            print(f"No valid labels found in {phase} set")
            return {"f1_macro": 0, "precision_macro": 0, "recall_macro": 0}
        
        valid_preds = [all_preds[i] for i in valid_indices]
        valid_labels = [all_labels[i] for i in valid_indices]
        
        # Calculate metrics
        f1_macro = f1_score(valid_labels, valid_preds, average='macro')
        precision_macro = precision_score(valid_labels, valid_preds, average='macro')
        recall_macro = recall_score(valid_labels, valid_preds, average='macro')
        
        # Calculate F1 for each class
        f1_per_class = f1_score(valid_labels, valid_preds, average=None)
        class_names = ['against', 'favor', 'neutral']
        for i, score in enumerate(f1_per_class):
            print(f"  F1 {class_names[i]}: {score:.4f}")
        
        # Print overall metrics
        print(f"  {phase.capitalize()} Metrics:")
        print(f"    F1 Macro: {f1_macro:.4f}")
        print(f"    Precision Macro: {precision_macro:.4f}")
        print(f"    Recall Macro: {recall_macro:.4f}")
        
        return {
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_per_class": f1_per_class
        }
    
    def predict(self, test_path, chunk_aggregation_method=None):
        """
        Run inference on test data and return predictions.
        
        Args:
            test_path: Path to the test data
            chunk_aggregation_method: Method to aggregate document chunks. If None, use self.chunk_aggregation_method.
                Options are 'max_pooling', 'simple_averaging', or 'weighted_averaging'.
        
        Returns:
            predictions_df: DataFrame with predictions for each chunk
            aggregated_df: DataFrame with aggregated predictions for each document
        """
        self.model.eval()
        
        # Load test data
        test_loader = split_finetuning_dataloader(
            csv_path=test_path,
            batch_size=self.args.batch_size*2,
            max_length=self.args.max_length,
            model=self.args.model,
            num_workers=self.args.n_workers,
            process_original=self.args.process_original,
            max_doc_length=self.args.max_doc_length,
            overlap=self.args.overlap,
            shuffle=False
        )
        
        # Initialize lists for storing results
        all_predictions = []
        
        # Inference loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Inference", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                # Get metadata
                speakers = batch['speaker']
                chambers = batch['chamber']
                targets = batch['target']
                subtopics_1 = batch['subtopic_1']
                subtopics_2 = batch['subtopic_2']
                documents = batch['document']
                document_ids = batch['document_id']
                chunks = batch['chunk']
                total_chunks = batch['total_chunks']
                
                # Forward pass
                logits = self.model(input_ids, attention_mask, token_type_ids)
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Convert to numpy arrays
                probs_np = probs.cpu().numpy()
                preds_np = preds.cpu().numpy()
                
                # Map numeric predictions to labels
                stance_labels = {0: 'AGAINST', 1: 'FAVOR', 2: 'NONE'}
                stance_text = [stance_labels[pred] for pred in preds_np]
                
                # Store results for each sample
                for i in range(len(preds_np)):
                    confidence = probs_np[i][preds_np[i]]
                    result = {
                        'document_id': document_ids[i],
                        'chunk': int(chunks[i]),
                        'total_chunks': int(total_chunks[i]),
                        'speaker': speakers[i],
                        'chamber': chambers[i],
                        'target': targets[i],
                        'subtopic_1': subtopics_1[i],
                        'subtopic_2': subtopics_2[i],
                        'document': documents[i],
                        'stance_prediction': int(preds_np[i]),
                        'stance': stance_text[i],
                        'stance_confidence': float(confidence),
                        'prob_against': float(probs_np[i][0]),
                        'prob_favor': float(probs_np[i][1]),
                        'prob_none': float(probs_np[i][2])
                    }
                    all_predictions.append(result)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        # Use the provided method if specified, otherwise use the instance variable
        if chunk_aggregation_method is not None:
            agg_method = chunk_aggregation_method
        else:
            agg_method = self.chunk_aggregation_method
            
        print(f"Running inference with chunk aggregation method: {agg_method}")
        
        # Aggregate chunks by document ID using the specified method
        if 'document_id' in predictions_df.columns and 'chunk' in predictions_df.columns:
            print(f"Aggregating chunks using {agg_method}...")
            
            # Extract document IDs and probability arrays
            doc_ids = predictions_df['document_id'].tolist()
            probs_list = [np.array([row['prob_against'], row['prob_favor'], row['prob_none']]) 
                         for _, row in predictions_df.iterrows()]
            
            # Get chunk texts for weighted averaging if needed
            chunk_texts = None
            if agg_method == 'weighted_averaging':
                chunk_texts = predictions_df['document'].tolist()
            
            # Combine chunks using the specified method
            doc_probs = combine_chunks(probs_list, doc_ids, chunk_texts, method=agg_method)
            
            # Create metadata for each document
            document_metadata = {}
            for _, row in predictions_df.iterrows():
                doc_id = row['document_id']
                if doc_id not in document_metadata:
                    document_metadata[doc_id] = {
                        'speaker': row['speaker'],
                        'chamber': row['chamber'],
                        'target': row['target'],
                        'subtopic_1': row['subtopic_1'],
                        'subtopic_2': row['subtopic_2'],
                        'document': row['document'],
                        'total_chunks': row['total_chunks']
                    }
            
            # Create aggregated results
            aggregated_results = []
            for doc_id, probs in doc_probs.items():
                # Get metadata
                meta = document_metadata[doc_id]
                
                # Get stance prediction from probabilities
                stance_pred = np.argmax(probs)
                confidence = probs[stance_pred]
                
                # Map to text label
                stance_text = stance_labels[stance_pred]
                
                # Find the original chunk with this probability array (for most_important_document_chunk)
                doc_chunks = predictions_df[predictions_df['document_id'] == doc_id]
                
                # If using max_pooling, find the chunk with the highest max probability
                if agg_method == 'max_pooling':
                    # Find the chunk with maximum probability in any class
                    max_values = [max(row['prob_against'], row['prob_favor'], row['prob_none']) 
                                 for _, row in doc_chunks.iterrows()]
                    max_chunk_idx = np.argmax(max_values)
                    important_chunk = doc_chunks.iloc[max_chunk_idx]['chunk']
                else:
                    # For averaging methods, use the chunk with the highest probability for the predicted stance
                    if stance_pred == 0:  # Against
                        important_chunk = doc_chunks.iloc[doc_chunks['prob_against'].argmax()]['chunk']
                    elif stance_pred == 1:  # Favor
                        important_chunk = doc_chunks.iloc[doc_chunks['prob_favor'].argmax()]['chunk']
                    else:  # None/neutral
                        important_chunk = doc_chunks.iloc[doc_chunks['prob_none'].argmax()]['chunk']
                
                # Create result
                result = {
                    'document_id': doc_id,
                    'speaker': meta['speaker'],
                    'chamber': meta['chamber'],
                    'target': meta['target'],
                    'subtopic_1': meta['subtopic_1'],
                    'subtopic_2': meta['subtopic_2'],
                    'document': meta['document'],
                    'document_count': 1,  # Always 1 for document-level aggregation
                    'chunk_count': int(meta['total_chunks']),
                    'stance_prediction': int(stance_pred),
                    'stance': stance_text,
                    'stance_confidence': float(confidence),
                    'prob_against': float(probs[0]),
                    'prob_favor': float(probs[1]),
                    'prob_none': float(probs[2]),
                    'most_important_document_chunk': int(important_chunk)
                }
                aggregated_results.append(result)
            
            # Convert to DataFrame
            aggregated_df = pd.DataFrame(aggregated_results)
            
            # Return both detailed and aggregated results
            return predictions_df, aggregated_df
        
        # If no document_id or chunk columns, return original predictions
        return predictions_df, predictions_df
