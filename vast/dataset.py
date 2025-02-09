#!/usr/bin/env python
"""
dataset.py

Defines dataset classes for:
1) VAST train/dev data (with labels).
2) Inference data (no labels, only document & topic).

Instead of combining document and topic into a single string, they are kept separate.
"""

import pandas as pd

class VASTTrainDataset:
    """
    Loads the VAST data for stance classification.
    Expects CSV columns:
      - post      : the main text/comment (document)
      - topic_str : the associated topic
      - label     : stance label (0=con, 1=pro, 2=neutral)
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.documents = self.data["post"].astype(str).tolist()
        self.topics = self.data["topic_str"].astype(str).tolist()
        self.labels = self.data["label"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "document": self.documents[idx],
            "topic": self.topics[idx],
            "label": self.labels[idx]
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
        # Drop rows where a key field (e.g. policy_area) is missing.
        self.df = self.df.dropna(subset=["policy_area"])
        # Group by document and take the first row (assumes same subtopic per document)
        self.df = (
            self.df.groupby("document", as_index=False)
            .first()
            .reset_index(drop=True)
        )
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
