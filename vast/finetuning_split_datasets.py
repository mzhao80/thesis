import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from split_long_documents import split_document_at_sentence_boundaries, estimate_combined_length
from split_datasets_subtopic import clean_data
os.environ['TOKENIZERS_PARALLELISM'] = '0'

class SplitFinetuningDataset(Dataset):
    def __init__(self, csv_path, max_length=512, model='sentence-transformers/all-mpnet-base-v2',
                 process_original=False, max_doc_length=384, overlap=50):
        """
        Dataset for fine-tuning on split documents with targets and stance labels.
        
        Args:
            csv_path: Path to the CSV file with documents, targets, and stance labels
            max_length: Maximum token length for encoding
            model: Model name for tokenizer
            process_original: Whether to split original documents or use pre-split data
            max_doc_length: Maximum document length when splitting (only used if process_original=True)
            overlap: Overlap between chunks when splitting (only used if process_original=True)
        """
        if process_original:
            # Load the original data and split documents
            print(f"Loading and splitting original data from {csv_path}")
            df = pd.read_csv(csv_path, index_col=False)
            df = clean_data(df)
            
            # Drop rows with missing document, target, or stance (label)
            initial_len = len(df)
            df = df.dropna(subset=['document', 'target', 'label'])
            dropped_len = initial_len - len(df)
            if dropped_len > 0:
                print(f"Dropped {dropped_len} rows with missing document, target, or stance")
            
            # Split the documents
            split_data = []
            
            from transformers import AutoTokenizer
            temp_tokenizer = AutoTokenizer.from_pretrained(model)
            
            for idx, row in df.iterrows():
                document = str(row['document']) if row['document'] is not None else ""
                target = str(row['target']) if row['target'] is not None else ""
                summary = str(row['summary']) if 'summary' in row and row['summary'] is not None else ""
                
                # Get the stance label
                label = int(row['label'])
                
                # Estimate token length to determine if splitting is needed
                input_format = f"text: {document} target: {target}"
                total_length = estimate_combined_length(document, target, summary, temp_tokenizer)
                
                if total_length <= max_length:
                    # Document is short enough, no need to split
                    split_data.append({
                        'document': document,
                        'speaker': row['speaker'] if 'speaker' in row else "",
                        'chamber': row['chamber'] if 'chamber' in row else "",
                        'target': target,
                        'summary': summary,
                        'subtopic_1': row['subtopic_1'] if 'subtopic_1' in row else "",
                        'subtopic_2': row['subtopic_2'] if 'subtopic_2' in row else "",
                        'document_id': f"{idx}",
                        'document_chunk': 0,
                        'document_chunks_total': 1,
                        'label': label
                    })
                else:
                    # Split the document
                    chunks = split_document_at_sentence_boundaries(document, temp_tokenizer, max_doc_length, overlap)
                    
                    for i, chunk in enumerate(chunks):
                        split_data.append({
                            'document': chunk,
                            'speaker': row['speaker'] if 'speaker' in row else "",
                            'chamber': row['chamber'] if 'chamber' in row else "",
                            'target': target,
                            'summary': summary,
                            'subtopic_1': row['subtopic_1'] if 'subtopic_1' in row else "",
                            'subtopic_2': row['subtopic_2'] if 'subtopic_2' in row else "",
                            'document_id': f"{idx}",
                            'document_chunk': i,
                            'document_chunks_total': len(chunks),
                            'label': label
                        })
            
            # Convert to DataFrame
            self.df = pd.DataFrame(split_data)
            print(f"Created {len(self.df)} document chunks from {len(df)} original documents")
        else:
            # Use pre-split data
            self.df = pd.read_csv(csv_path, index_col=False)
            self.df = clean_data(self.df)
            print(f"Loaded {len(self.df)} document chunks")
        
        # Check for required columns
        required_columns = ['document', 'target', 'label']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            error_message = f"Error: Required columns missing from dataset: {missing_columns}"
            print(error_message)
            raise ValueError(error_message)
            
        # Extract relevant columns
        self.documents = self.df['document'].tolist()
        self.speakers = self.df['speaker'].tolist() if 'speaker' in self.df.columns else [""] * len(self.documents)
        self.chambers = self.df['chamber'].tolist() if 'chamber' in self.df.columns else [""] * len(self.documents)
        
        # Use labels as integers directly
        self.labels = self.df['label'].tolist()
        
        # Set up targets and summaries
        self.targets = self.df['target'].tolist() if 'target' in self.df.columns else self.df['policy_area'].tolist()
        self.summaries = []
        
        for i, row in self.df.iterrows():
            if 'summary' in row and not pd.isna(row['summary']):
                self.summaries.append(row['summary'])
            else:
                self.summaries.append("")
                
        # Check for null values in required columns
        null_docs = sum(pd.isna(self.df['document']))
        null_targets = sum(pd.isna(self.df['target']))
        null_labels = sum(pd.isna(self.df['label']))
        
        if null_docs > 0 or null_targets > 0 or null_labels > 0:
            error_message = f"Error: Found null values in required columns - documents: {null_docs}, targets: {null_targets}, labels: {null_labels}"
            print(error_message)
            raise ValueError(error_message)
        
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
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_token_type_ids=True
        )
        
        # Convert labels to tensor
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)
        
        # Get chunk metadata
        self.chunks = self.df['document_chunk'].tolist() if 'document_chunk' in self.df.columns else [0] * len(self.documents)
        self.total_chunks = self.df['document_chunks_total'].tolist() if 'document_chunks_total' in self.df.columns else [1] * len(self.documents)
        
        # Store original document indices for aggregation later
        if 'document_id' in self.df.columns:
            self.document_ids = self.df['document_id'].tolist()
        else:
            # If no document_id column exists, we'll create unique IDs based on speaker and target
            self.document_ids = []
            for idx, row in self.df.iterrows():
                speaker = row['speaker'] if 'speaker' in row else ""
                target = row['target'] if 'target' in row else ""
                chunk = row['document_chunk'] if 'document_chunk' in self.df.columns else 0
                doc_id = f"{speaker}_{target}_{idx // (chunk+1)}"
                self.document_ids.append(doc_id)
        
        if 'subtopic_1' in self.df.columns:
            self.subtopics_1 = self.df['subtopic_1'].tolist()
        else:
            self.subtopics_1 = [""] * len(self.documents)
            
        if 'subtopic_2' in self.df.columns:
            self.subtopics_2 = self.df['subtopic_2'].tolist()
        else:
            self.subtopics_2 = [""] * len(self.documents)
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': self.labels_tensor[idx],
            'speaker': self.speakers[idx],
            'target': self.targets[idx],
            'summary': self.summaries[idx],
            'subtopic_1': self.subtopics_1[idx],
            'subtopic_2': self.subtopics_2[idx],
            'document': self.documents[idx],
            'chamber': self.chambers[idx],
            'document_id': self.document_ids[idx],
            'chunk': self.chunks[idx],
            'total_chunks': self.total_chunks[idx]
        }
        return item


def split_finetuning_dataloader(csv_path, batch_size=8, max_length=512, 
                               model='sentence-transformers/all-mpnet-base-v2', 
                               num_workers=4, process_original=False,
                               max_doc_length=384, overlap=50,
                               shuffle=True):
    """
    Create a DataLoader for fine-tuning on split documents.
    
    Args:
        csv_path: Path to the CSV file with documents to fine-tune on
        batch_size: Batch size for DataLoader
        max_length: Maximum token length for encoding
        model: Model name for tokenizer
        num_workers: Number of workers for DataLoader
        process_original: Whether to split original documents or use pre-split data
        max_doc_length: Maximum document length when splitting
        overlap: Overlap between chunks when splitting
        shuffle: Whether to shuffle the data
    """
    dataset = SplitFinetuningDataset(
        csv_path=csv_path, 
        max_length=max_length, 
        model=model,
        process_original=process_original,
        max_doc_length=max_doc_length,
        overlap=overlap
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def create_train_val_test_split(csv_path, output_dir, train_ratio=0.8, val_ratio=0.1,
                               process_original=False, max_doc_length=384, overlap=50,
                               model='sentence-transformers/all-mpnet-base-v2', seed=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        csv_path: Path to the CSV file with documents
        output_dir: Directory to save the split datasets
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        process_original: Whether to split original documents
        max_doc_length: Maximum document length when splitting
        overlap: Overlap between chunks when splitting
        model: Model name for tokenizer
        seed: Random seed for consistent splits
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, index_col=False)
    print(f"Loaded {len(df)} rows")
    
    # Check for required columns
    required_columns = ['document', 'target', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_message = f"Error: Required columns missing from dataset: {missing_columns}"
        print(error_message)
        raise ValueError(error_message)
    
    # Check for null values
    null_docs = df['document'].isna().sum()
    null_targets = df['target'].isna().sum()
    null_labels = df['label'].isna().sum()
    
    if null_docs > 0 or null_targets > 0 or null_labels > 0:
        error_message = f"Error: Found null values in required columns - documents: {null_docs}, targets: {null_targets}, labels: {null_labels}"
        print(error_message)
        raise ValueError(error_message)
    
    if process_original:
        # Load the original data and split documents
        print(f"Loading and splitting original data from {csv_path}")
        df = pd.read_csv(csv_path, index_col=False)
        df = clean_data(df)
        
        # Split the documents
        split_data = []
        
        from transformers import AutoTokenizer
        temp_tokenizer = AutoTokenizer.from_pretrained(model)

        for idx, row in df.iterrows():
            document = str(row['document']) if row['document'] is not None else ""
            target = str(row['target']) if row['target'] is not None else ""
            summary = str(row['summary']) if 'summary' in row and row['summary'] is not None else ""
            
            # Get the stance label
            label = int(row['label']) 
            
            # Estimate token length to determine if splitting is needed
            total_length = estimate_combined_length(document, target, summary, temp_tokenizer)
            
            if total_length <= 512:  # Typical max length for BERT models
                # Document is short enough, no need to split
                split_data.append({
                    'document': document,
                    'speaker': row['speaker'] if 'speaker' in row else "",
                    'chamber': row['chamber'] if 'chamber' in row else "",
                    'target': target,
                    'summary': summary,
                    'subtopic_1': row['subtopic_1'] if 'subtopic_1' in row else "",
                    'subtopic_2': row['subtopic_2'] if 'subtopic_2' in row else "",
                    'document_id': f"{idx}",
                    'document_chunk': 0,
                    'document_chunks_total': 1,
                    'label': label
                })
            else:
                # Split the document
                chunks = split_document_at_sentence_boundaries(document, temp_tokenizer, max_doc_length, overlap)
                
                for i, chunk in enumerate(chunks):
                    split_data.append({
                        'document': chunk,
                        'speaker': row['speaker'] if 'speaker' in row else "",
                        'chamber': row['chamber'] if 'chamber' in row else "",
                        'target': target,
                        'summary': summary,
                        'subtopic_1': row['subtopic_1'] if 'subtopic_1' in row else "",
                        'subtopic_2': row['subtopic_2'] if 'subtopic_2' in row else "",
                        'document_id': f"{idx}",
                        'document_chunk': i,
                        'document_chunks_total': len(chunks),
                        'label': label
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(split_data)
        print(f"Created {len(df)} document chunks from {len(df)} original documents")
    else:
        # Use pre-split data
        df = pd.read_csv(csv_path)
        df = clean_data(df)
        print(f"Loaded {len(df)} document chunks")
    
    # Group document IDs by label to ensure stratified sampling
    doc_ids_by_label = {}
    
    for idx, row in df.iterrows():
        label = int(row['label'])
        if label not in doc_ids_by_label:
            doc_ids_by_label[label] = []
        doc_ids_by_label[label].append(idx)
    
    # Print label distribution
    print("Label distribution in original dataset:")
    for label, doc_ids in doc_ids_by_label.items():
        label_name = {0: "con/against", 1: "pro/favor", 2: "neutral"}[label]
        print(f"  Label {label} ({label_name}): {len(doc_ids)} documents ({len(doc_ids)/len(df)*100:.1f}%)")
    
    # Set random seed for reproducible splits
    np.random.seed(seed)
    
    # Shuffle document IDs within each label group
    for label in doc_ids_by_label:
        np.random.shuffle(doc_ids_by_label[label])
    
    # Create stratified splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, indices in doc_ids_by_label.items():
        n_samples = len(indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])
    
    # Create the split dataframes
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # Print the stratified split statistics
    print("\nAfter stratified splitting:")
    print(f"Train set: {len(train_df)} documents")
    print(f"Val set: {len(val_df)} documents")
    print(f"Test set: {len(test_df)} documents")
    
    # Print label distribution in each split
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name} set label distribution:")
        label_counts = split_df['label'].value_counts().to_dict()
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            label_name = {0: "con/against", 1: "pro/favor", 2: "neutral"}[label]
            print(f"  Label {label} ({label_name}): {count} documents ({count/len(split_df)*100:.1f}%)")
    
    # Save the dataframes
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved {len(train_df)} training examples to {train_path}")
    print(f"Saved {len(val_df)} validation examples to {val_path}")
    print(f"Saved {len(test_df)} test examples to {test_path}")
    
    return train_path, val_path, test_path
