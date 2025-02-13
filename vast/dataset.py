#!/usr/bin/env python
"""
dataset.py

Defines dataset classes for:
1) VAST train/dev data (with labels).
2) Inference data (no labels, only document & topic).

Instead of combining document and topic into a single string, they are kept separate.
"""

import pandas as pd
from torch.utils.data import Dataset

class VASTTrainDataset:
    """
    Loads the VAST data for stance classification.
    Expects CSV columns:
      - post      : the main text/comment (document)
      - new_topic : the associated topic
      - label     : stance label (0=con, 1=pro, 2=neutral)
      - seen      : zero-shot (0) or few-shot (1)
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.documents = self.data["post"].astype(str).tolist()
        self.topics = self.data["topic_str"].astype(str).tolist()
        self.labels = self.data["label"].tolist()
        self.seen = self.data["seen?"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "document": self.documents[idx],
            "topic": self.topics[idx],
            "label": self.labels[idx],
            "seen": self.seen[idx]
        }

class InferenceDataset:
    """
    Loads data for inference (no labels).
    Expects CSV columns:
      - document : the main text
      - subtopic : the topic or subtopic
    """
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # Lowercase the subtopic.
        self.df["subtopic"] = self.df["subtopic"].str.lower()
        self.documents = self.df["document"].astype(str).tolist()
        self.topics = self.df["subtopic"].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "document": self.documents[idx],
            "topic": self.topics[idx]
        }

class EmbeddingDataset(Dataset):
    """
    A dataset class that holds pre-computed embeddings, labels, and the original document and topic.
    This is useful for debugging and for training the classifier.
    """
    def __init__(self, embeddings, labels, documents, topics, seen):
        self.embeddings = embeddings
        self.labels = labels
        self.documents = documents
        self.topics = topics
        self.seen = seen

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Combine document and topic for debug printing.
        combined_text = self.documents[idx] + " || " + self.topics[idx]
        return self.embeddings[idx], self.labels[idx], combined_text