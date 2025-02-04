import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

class DocTopicPhraseDataset(Dataset):
    """
    Dataset for document–policy_area–subtopic–phrase triples.
    Expects a CSV with columns: "document", "policy_area", "subtopic", "phrase".
    For this training, we condition on the policy area.
    
    This version groups by document and samples the first row per document,
    ensuring that each document appears exactly once.
    """
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # Drop rows with missing policy areas.
        self.df = self.df.dropna(subset=["policy_area"])
        # Group by document and sample one row from each group.
        # It is guaranteed that each document has the same policy area and subtopic.
        self.df = (
            self.df.groupby("document", as_index=False)
            .first()
            .reset_index(drop=True)
        )
        # Use the NV-Embed-v2 tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
        # Build mapping from policy_area to index.
        unique_parents = self.df["policy_area"].unique().tolist()
        self.topic_to_index = {topic: idx for idx, topic in enumerate(unique_parents)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        document = row["document"]
        policy = row["policy_area"]
        # remove leading and trailing quotation marks
        phrase = row["phrase"].strip('"')
        
        # Tokenize without truncation or padding.
        doc_enc = self.tokenizer(document, padding=False, truncation=False, return_tensors="pt")
        phrase_enc = self.tokenizer(phrase, padding=False, truncation=False, return_tensors="pt")
        
        topic_idx = self.topic_to_index.get(policy)
        if topic_idx is None:
            print(idx, document, policy, phrase)
            topic_idx = len(self.topic_to_index)
            self.topic_to_index[policy] = topic_idx
        
        return {
            "document": document,  # raw text (for the SentenceTransformer)
            "doc_input_ids": doc_enc["input_ids"].squeeze(0),
            "doc_attention_mask": doc_enc["attention_mask"].squeeze(0),
            "phrase": phrase,  # raw text for debugging
            "phrase_input_ids": phrase_enc["input_ids"].squeeze(0),
            "topic_idx": topic_idx,
            "policy": policy
        }

def collate_fn_with_tokenizer(tokenizer):
    """
    Returns a collate function that vectorizes a batch.
    
    Since each document now has only one candidate phrase,
    we simply pad the document input_ids, attention masks, and phrase input_ids.
    """
    from torch.nn.utils.rnn import pad_sequence
    def _collate_fn(batch):
        docs = [item["document"] for item in batch]
        doc_input_ids_list = [item["doc_input_ids"] for item in batch]
        doc_attention_mask_list = [item["doc_attention_mask"] for item in batch]
        phrase_input_ids_list = [item["phrase_input_ids"] for item in batch]
        topic_idxs = torch.tensor([item["topic_idx"] for item in batch], dtype=torch.long)
        policies = [item["policy"] for item in batch]
        
        docs_input_ids = pad_sequence(doc_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        docs_attention_mask = pad_sequence(doc_attention_mask_list, batch_first=True, padding_value=0)
        phrase_input_ids = pad_sequence(phrase_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        return docs, docs_input_ids, docs_attention_mask, phrase_input_ids, topic_idxs, policies
    return _collate_fn
