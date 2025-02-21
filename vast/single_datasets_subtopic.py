import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
os.environ['TOKENIZERS_PARALLELISM'] = '0'

class VASTSubtopicDataset(Dataset):
    def __init__(self, model='bert-base', wiki_model=''):
        # Read the taxonomy data
        df = pd.read_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv')
        print(f'# of total examples: {df.shape[0]}')
        
        # Drop rows where required fields are missing
        df = df.dropna(subset=['speaker', 'document', 'summary', 'subtopic_2'])
        print(f'# of examples after dropping na: {df.shape[0]}')

        # Store the data
        self.speakers = df['speaker'].tolist()
        self.documents = df['document'].tolist()
        self.subtopics = df['subtopic_2'].tolist()
        self.stances = [-1] * len(self.documents)  # No labels for inference
        
        # Load the tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Process text data
        self.encodings = self.tokenizer(
            self.documents,
            self.subtopics,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
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
            'subtopic': self.subtopics[idx],
            'document': self.documents[idx]
        }
        return item

def subtopic_data_loader(batch_size, model='bert-base', wiki_model='', n_workers=4):
    dataset = VASTSubtopicDataset(model=model, wiki_model=wiki_model)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True
    )
    
    return loader
