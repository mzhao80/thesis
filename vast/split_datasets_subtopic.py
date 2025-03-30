"""
StrideStance Dataset Utilities for Split Documents with Subtopics

This module implements dataset loading and processing functionality for the StrideStance
approach, specifically for handling split documents organized by subtopics. It focuses on:

1. Loading and preprocessing split document chunks with their metadata
2. Creating PyTorch DataLoaders for efficient batch processing
3. Integrating document chunks with their corresponding subtopic information
4. Preparing input data in the format required by the stance detection models

Key features:
- Support for processing document chunks with preserved metadata
- Integration of knowledge-enhanced features for stance detection
- Batch creation with appropriate padding and tokenization
- Tracking of document-chunk relationships for post-processing
- Support for handling subtopic hierarchies

This dataset module is essential for feeding properly formatted data to the StrideStance
models during both training and inference, while maintaining the relationships between
document chunks, original documents, and subtopics.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
os.environ['TOKENIZERS_PARALLELISM'] = '0'

def clean_data(df):
    # if speaker is carolyn b exactly, replace with carolyn b maloney
    df['speaker'] = df['speaker'].replace('CAROLYN B', 'CAROLYN B. MALONEY')
    # replace LOIS FRANKEL with FRANKEL
    df['speaker'] = df['speaker'].replace('Ms. LOIS FRANKEL of Florida', 'Ms. FRANKEL of Florida')
    # replace "Ms JACKSON LEE" with "Ms. JACKSON LEE"
    df['speaker'] = df['speaker'].replace('Ms. JACKSON LEE', 'Ms. JACKSON LEE')
    # replace Ms. BORDEAUX with Ms. BOURDEAUX
    df['speaker'] = df['speaker'].replace('Ms. BORDEAUX', 'Ms. BOURDEAUX')
    # replace Mr. SABAN with Mr. SABLAN
    df['speaker'] = df['speaker'].replace('Mr. SABAN', 'Mr. SABLAN')
    # replace Ms. JOHNSON  of Texas with Ms. JOHNSON of Texas
    df['speaker'] = df['speaker'].replace('Ms. JOHNSON  of Texas', 'Ms. JOHNSON of Texas')
    # replace SEMPOLINKSI with SEMPOLINSKI
    df['speaker'] = df['speaker'].replace('Mr. SEMPOLINKSI', 'Mr. SEMPOLINSKI')

    # replace "Miss" with "Ms."
    df['speaker'] = df['speaker'].replace("Miss GONZALEZ-COLON", "Ms. GONZALEZ-COLON")
    df['speaker'] = df['speaker'].replace("Miss RICE of New York", "Ms. RICE of New York")

    # drop where speaker is The VICE PRESIDENT
    df = df[df['speaker'] != 'The VICE PRESIDENT']
    
    return df

class SplitVASTSubtopicDataset(Dataset):
    def __init__(self, csv_path, max_length=512, model='sentence-transformers/all-mpnet-base-v2'):
        """
        Dataset for handling split documents with targets and summaries.
        
        Args:
            csv_path: Path to the CSV file with split documents
            max_length: Maximum token length for encoding (default: 512)
            model: Model name for tokenizer
        """
        self.df = pd.read_csv(csv_path)
        print("Original length: ", len(self.df))
        self.df = self.df.dropna(subset=['target', 'document', 'speaker', 'subtopic_1', 'subtopic_2'])
        print("After dropping na: ", len(self.df))
        self.df = clean_data(self.df)
        print(f"Loaded {len(self.df)} document chunks")
        
        # Extract relevant columns
        self.documents = self.df['document'].tolist()
        self.speakers = self.df['speaker'].tolist()
        
        if 'label' in self.df.columns:
            self.stances = self.df['label'].tolist()
        else:
            self.stances = [-1] * len(self.documents)  # Default to -1 for inference
            
        if 'chamber' in self.df.columns:
            self.chambers = self.df['chamber'].tolist()
        else:
            self.chambers = [''] * len(self.documents)
        
        # Set up targets and summaries
        self.targets = self.df['target'].tolist()
        self.summaries = self.df['summary'].tolist() if 'summary' in self.df.columns else [''] * len(self.documents)
        self.subtopics_1 = self.df['subtopic_1'].tolist()
        self.subtopics_2 = self.df['subtopic_2'].tolist()
        
        # Get chunk metadata
        self.chunks = self.df['document_chunk'].tolist() if 'document_chunk' in self.df.columns else [1] * len(self.documents)
        self.total_chunks = self.df['document_chunks_total'].tolist() if 'document_chunks_total' in self.df.columns else [1] * len(self.documents)
        
        # Store original document indices for aggregation later
        if 'document_id' in self.df.columns:
            self.document_ids = self.df['document_id'].tolist()
        else:
            # If no document_id column exists, we'll create unique IDs based on speaker and target
            self.document_ids = []
            for idx, row in self.df.iterrows():
                speaker = row['speaker']
                target = row['target']
                chunk = row['document_chunk'] if 'document_chunk' in self.df.columns else 1
                doc_id = f"{speaker}_{target}_{idx // chunk}"
                self.document_ids.append(doc_id)
        
        # Load the tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        print("Using format: text: document target: target + summary")
        # Create "text: {document} target: {target}" format
        text_targets = []
        for doc, target in zip(self.documents, self.targets):
            # Ensure both document and target are strings
            doc = str(doc) if doc is not None else ""
            target = str(target) if target is not None else ""
            text_targets.append(f"text: {doc} target: {target}")
        
        # Ensure summaries are valid strings
        processed_summaries = []
        for summary in self.summaries:
            summary = str(summary) if summary is not None else ""
            processed_summaries.append(summary)
        
        # This tokenizes as [CLS] text_target [SEP] summary [SEP]
        self.encodings = self.tokenizer(
            text_targets,
            processed_summaries,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_token_type_ids=True
        )
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': self.stances[idx],
            'speaker': self.speakers[idx],
            'target': self.targets[idx],
            'summary': self.summaries[idx],
            'subtopic_1': self.subtopics_1[idx],
            'subtopic_2': self.subtopics_2[idx],
            'document': self.documents[idx],
            'chamber': self.chambers[idx],
            'document_id': self.document_ids[idx],
            'chunk': int(self.chunks[idx]),  # Convert to int to avoid dtype issues
            'total_chunks': int(self.total_chunks[idx])  # Convert to int to avoid dtype issues
        }
        return item

def custom_collate_fn(batch):
    """
    Custom collate function that handles mixed tensor and non-tensor fields.
    Args:
        batch: List of data items from the dataset
    Returns:
        Batched data with appropriate types
    """
    elem = batch[0]
    batch_dict = {}
    
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            # These are tensors and should be stacked
            batch_dict[key] = torch.stack([d[key] for d in batch])
        else:
            # These are strings, ints, or other non-tensor fields and should remain as lists
            batch_dict[key] = [d[key] for d in batch]
    
    return batch_dict


def split_subtopic_dataloader(csv_path, batch_size=8, max_length=512, model='sentence-transformers/all-mpnet-base-v2', num_workers=4):
    """
    Create a DataLoader for split documents with targets and summaries.
    
    Args:
        csv_path: Path to the CSV file with split documents
        batch_size: Batch size for DataLoader
        max_length: Maximum token length for encoding
        model: Model name for tokenizer
        num_workers: Number of workers for DataLoader
    """
    dataset = SplitVASTSubtopicDataset(csv_path, max_length, model)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # Use our custom collate function
    )
    
    return dataloader
