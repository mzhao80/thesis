#!/usr/bin/env python
"""
dataset.py

Defines dataset classes for stance detection with Wikipedia enrichment.
For each topic, Wikipedia data (summary or full content) can be used, depending on the setting.
"""

import pandas as pd
from mediawiki import MediaWiki
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import hashlib

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

class WikiDataset:
    """
    Loads the VAST training data with optional Wikipedia background.
    Expects CSV columns:
      - post      : the main text/comment (document)
      - new_topic : the associated topic
      - label     : stance label (0=con, 1=pro, 2=neutral)
      - seen?     : zero-shot (0) or few-shot (1)
    
    The Wikipedia data is cached in a CSV file with columns:
      - topic     : the topic string
      - query     : the suggested Wikipedia query from wikipedia.suggest()
      - summary   : the Wikipedia page summary
      - content   : the full Wikipedia page content
    
    Based on wiki_mode, returns either:
      - none    : empty string for wiki text
      - summary : just the Wikipedia summary
      - content : full Wikipedia page content
    
    For each item, returns (document, topic, query, wiki_text, label, seen)
    """
    def __init__(self, csv_file, wiki_mode="none"):
        self.data = pd.read_csv(csv_file)
        # lowercase topics
        self.data["new_topic"] = self.data["new_topic"].str.lower()
        self.data["topic_str"] = self.data["topic_str"].str.lower()
        self.documents = self.data["post"].astype(str).tolist()
        self.topics = self.data["topic_str"].astype(str).tolist()
        self.labels = self.data["label"].tolist()
        self.seen = self.data["seen?"].tolist()
        self.new_topics = self.data["new_topic"].astype(str).tolist()
        
        if wiki_mode == "none":
            # Use topic name for both query and wiki text in none mode
            self.wiki_texts = self.topics
            self.queries = self.topics
            return
            
        # Generate cache filename based on input file
        cache_file = "vast_wiki_cache.csv"
        
        # Initialize cache data
        cache_data = {
            'topic': [],
            'query': [],
            'summary': [],
            'content': []
        }
        
        # Try to load existing cache first
        if os.path.exists(cache_file):
            print(f"Loading existing Wikipedia cache: {cache_file}")
            cache_df = pd.read_csv(cache_file)
            for _, row in cache_df.iterrows():
                cache_data['topic'].append(row['topic'])
                cache_data['query'].append(row['query'])
                cache_data['summary'].append(row['summary'])
                cache_data['content'].append(row['content'])
        
        wikipedia = MediaWiki()
        
        # Fetch missing topics
        for new_topic, topic in tqdm(zip(self.new_topics, self.topics), desc="Fetching Wikipedia data"):
            if new_topic not in cache_data['topic']:
                try:
                    #suggested_query = wikipedia.suggest(topic)
                    #print(f"Topic: {topic}, Suggested Query: {suggested_query}")
                    
                    #if suggested_query is None:
                        #raise Exception("No Wikipedia suggestion found")
                        
                    summary = wikipedia.summary(new_topic, auto_suggest=False)
                    page = wikipedia.page(new_topic, auto_suggest=False)
                    content = page.content
                    
                    cache_data['topic'].append(topic)
                    cache_data['query'].append(new_topic)
                    cache_data['summary'].append(summary)
                    cache_data['content'].append(content)
                    
                except Exception as e:
                    print(f"Error fetching Wikipedia data for topic '{topic}': {str(e)}")
                    cache_data['topic'].append(topic)
                    cache_data['query'].append(new_topic)  # Use topic as query when wiki lookup fails
                    cache_data['summary'].append(new_topic)  # Use topic as summary when wiki lookup fails
                    cache_data['content'].append(new_topic)  # Use topic as content when wiki lookup fails
        
        # Save complete cache
        cache_df = pd.DataFrame(cache_data)
        cache_df.to_csv(cache_file, index=False)
        print(f"Wikipedia cache saved to: {cache_file}")
        
        # Create mappings for topics to queries and wiki texts
        topic_to_query = dict(zip(cache_df['topic'], cache_df['query']))
        topic_to_wiki = dict(zip(cache_df['topic'], 
                               cache_df['summary'] if wiki_mode == "summary" else cache_df['content']))
        
        # Get wiki texts and queries in the same order as topics
        self.wiki_texts = [topic_to_wiki.get(topic, topic) for topic in self.topics]  # Fallback to topic
        self.queries = [topic_to_query.get(topic, topic) for topic in self.topics]  # Fallback to topic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"document": self.documents[idx], 
                "topic": self.topics[idx],
                "query": self.queries[idx], 
                "wiki_text": self.wiki_texts[idx], 
                "label": self.labels[idx], 
                "seen": self.seen[idx]}

class EmbeddingDataset(Dataset):
    """
    A dataset class that holds pre-computed embeddings, labels, and extra information.
    """
    def __init__(self, embeddings, labels, documents, topics, seen):
        self.embeddings = embeddings
        self.labels = labels
        self.documents = documents
        self.topics = topics
        self.seen = seen

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        extra_info = {
            "document": self.documents[idx],
            "topic": self.topics[idx],
            "seen": self.seen[idx]
        }
        return self.embeddings[idx], self.labels[idx], extra_info
