import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
os.environ['TOKENIZERS_PARALLELISM'] = '0'

def clean_data(df):
    # if speaker is "Mr. ESHOO" replace with "Ms. ESHOO"
    df['speaker'] = df['speaker'].replace('Mr. ESHOO', 'Ms. ESHOO')
    # if speaker is "Mrs. CAROLYN B. MALONEY of New York" replace with "Ms. CAROLYN B. Maloney of New York"
    df['speaker'] = df['speaker'].replace('Mrs. CAROLYN B. MALONEY of New York', 'Ms. CAROLYN B. Maloney of New York')
    # replace Mrs. CAROLYN B with Ms. CAROLYN B. MALONEY of New York
    df['speaker'] = df['speaker'].replace('Mrs. CAROLYN B.', 'Ms. CAROLYN B. MALONEY of New York')
    # replace "Ms JACKSON LEE" with "Ms. JACKSON LEE"
    df['speaker'] = df['speaker'].replace('Ms. JACKSON LEE', 'Ms. JACKSON LEE')
    # replace Ms. BORDEAUX with Ms. BOURDEAUX
    df['speaker'] = df['speaker'].replace('Ms. BORDEAUX', 'Ms. BOURDEAUX')
    # replace Mr. SWALWELL of California with Mr. SWALWELL
    df['speaker'] = df['speaker'].replace('Mr. SWALWELL of California', 'Mr. SWALWELL')
    # replace Mr. SABAN with Mr. SABLAN
    df['speaker'] = df['speaker'].replace('Mr. SABAN', 'Mr. SABLAN')
    # replace Ms. JOHNSON  of Texas with Ms. Johnson of Texas
    df['speaker'] = df['speaker'].replace('Ms. JOHNSON  of Texas', 'Ms. Johnson of Texas')

    return df

class VASTSubtopicDataset(Dataset):
    def __init__(self, model='bert-base', wiki_model=''):
        # Read the taxonomy data
        df = pd.read_csv('/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv')
        df = clean_data(df)
        # filter df where chamber is "S"
        #df = df[df['chamber'] == 'S']
        print(f'# of total examples: {df.shape[0]}')
        
        # Drop rows where required fields are missing, but preserve indices
        df = df.dropna(subset=['speaker', 'document', 'target'])
        print(f'# of examples after dropping na: {df.shape[0]}')
        
        #only examine df where policy_area is Economics and Public Finance
        #df = df[df['policy_area'] != 'Health']
        #print(f'# of examples after filtering by policy_area: {df.shape[0]}')

        # Store the data
        self.speakers = df['speaker'].tolist()
        self.documents = df['document'].tolist()
        self.subtopics_1 = df['subtopic_1'].tolist()
        self.subtopics_2 = df['subtopic_2'].tolist()
        self.stances = [-1] * len(self.documents)  # No labels for inference
        self.chambers = df['chamber'].tolist()
        
        # Load the tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Process text data
        self.encodings = self.tokenizer(
            self.documents,
            self.subtopics_2,
            truncation=True,
            max_length=512,
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
            'subtopics_1': self.subtopics_1[idx],
            'subtopics_2': self.subtopics_2[idx],
            'document': self.documents[idx],
            'chamber': self.chambers[idx]
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
