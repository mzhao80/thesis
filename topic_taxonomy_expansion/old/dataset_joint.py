import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class DocTopicPhraseDataset(Dataset):
    """
    Dataset for document–parent topic–subtopic phrase triples.
    
    Expects a CSV with the following columns:
      - "document": Full text of a document.
      - "policy_area": The parent topic.
      - "phrase": The target subtopic phrase.
    """
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # Ensure proper string formatting.
        self.df["policy_area"] = self.df["policy_area"].astype(str)
        self.df["document"] = self.df["document"].astype(str)
        self.df["phrase"] = self.df["phrase"].astype(str)
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        document = row["document"]
        parent_topic = row["policy_area"]
        phrase = row["phrase"].strip('"')
        phrase_enc = self.tokenizer(phrase, padding=False, truncation=False, return_tensors="pt")
        return {
            "document": document,
            "parent_topic": parent_topic,
            "phrase_input_ids": phrase_enc["input_ids"].squeeze(0)
        }

def collate_fn(batch):
    """
    Collate function that:
      - Collects documents and parent topics as lists of strings.
      - Pads the phrase input ID sequences.
    """
    documents = [item["document"] for item in batch]
    parent_topics = [item["parent_topic"] for item in batch]
    phrase_input_ids_list = [item["phrase_input_ids"] for item in batch]
    phrase_input_ids = pad_sequence(phrase_input_ids_list, batch_first=True, padding_value=0)
    return documents, parent_topics, phrase_input_ids
