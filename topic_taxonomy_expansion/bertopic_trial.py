#!/usr/bin/env python
"""
BERTopic-based Taxonomy Generation

This script implements an alternative topic modeling approach using BERTopic to create 
a hierarchical taxonomy. BERTopic combines transformer-based embeddings with clustering
to create coherent topics and automatically organizes them into a hierarchical structure.

The script:
1. Loads and preprocesses training data
2. Initializes and configures a BERTopic model with appropriate parameters
3. Fits the model to extract topics and their hierarchical relationships
4. Organizes documents into a taxonomy based on their topic assignments
5. Outputs the taxonomy as a CSV file

Unlike the standard clustering approach or the hLDA approach, BERTopic uses modern
transformer-based models to create more semantically meaningful topic representations.

Input:
- Training data CSV from the lab directory

Output:
- bertopic_taxonomy.csv: Contains the hierarchical taxonomy with policy areas, subtopics,
  and source document indices
- bertopic_model: Saved BERTopic model for future reference
"""

import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
import re
import nltk
from nltk.corpus import stopwords
from scipy.cluster import hierarchy as sch

# Download NLTK resources (commented out as they only need to be run once)
#nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text for better topic modeling by removing noise and standardizing format.
    
    Args:
        text (str): The input text to preprocess
    
    Returns:
        str: The preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    """
    Main function to run the BERTopic analysis and generate the taxonomy.
    """
    # Define paths
    input_file = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv'
    output_file = 'bertopic_taxonomy.csv'
    
    print("Loading and preprocessing data...")
    # Load data
    df = pd.read_csv(input_file)
    df.fillna("", inplace=True)
    
    # Use document text for topic modeling
    df['combined_text'] = df['document']
    
    # Preprocess text
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    # Remove rows with empty processed text
    df = df[df['processed_text'].str.strip() != ""]
    df = df.reset_index(drop=True)
    
    # Create index mapping to keep track of original indices
    index_mapping = {i: df.index[i] for i in range(len(df))}
    
    documents = df['processed_text'].tolist()
    
    print("Initializing BERTopic model...")
    # Set up the embedding model - using a lightweight but effective model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Set up the vectorizer with stop words removal for cleaner topics
    stop_words = list(stopwords.words('english'))
    vectorizer = CountVectorizer(stop_words=stop_words)
    
    # Set up dimensionality reduction with UMAP
    # Parameters control the balance between local and global structure
    umap_model = UMAP(
        n_neighbors=15,     # Controls how local/global the embeddings are
        n_components=5,     # Reduced dimensionality
        metric='cosine',    # Semantic similarity measure
        random_state=42     # For reproducibility
    )
    
    # Set up clustering model with HDBSCAN
    # min_cluster_size controls the granularity of topics
    hdbscan_model = HDBSCAN(
        min_cluster_size=10  # Minimum documents to form a topic
    )
    
    # Create BERTopic model with the configured components
    topic_model = BERTopic(
        embedding_model=embedding_model,  # For document embeddings
        umap_model=umap_model,            # For dimensionality reduction
        hdbscan_model=hdbscan_model,      # For clustering
        vectorizer_model=vectorizer,      # For topic representation
        verbose=True                      # Show progress
    )
    
    print("Fitting BERTopic model...")
    # Fit the model and transform documents to get topic assignments
    topics, probs = topic_model.fit_transform(documents)
    
    print("Creating hierarchical topics...")
    # Create hierarchical representation using centroid linkage
    # This creates a tree structure of the topics
    linkage_function = lambda x: sch.linkage(x, 'centroid', optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(documents, linkage_function=linkage_function)
    
    # The hierarchical_topics structure has the following columns:
    # Index(['Parent_ID', 'Parent_Name', 'Topics', 'Child_Left_ID',
    #       'Child_Left_Name', 'Child_Right_ID', 'Child_Right_Name', 'Distance']
    
    # Create a parent-child relationship dictionary
    child_parent_map = {}
    name_map = {}
    for idx, row in hierarchical_topics.iterrows():
        parent_name = row['Parent_Name']
        name_map[idx] = parent_name
    
    # Get topic information including labels and counts
    topic_info = topic_model.get_topic_info()
    
    # Map documents to their topics and assign paths
    doc_paths = {}
    doc_topic_levels = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Assign each document to its detected topic and organize by hierarchy
    for doc_idx, topic_id in enumerate(topics):
        if topic_id == -1:  # Skip outliers
            doc_paths[doc_idx] = [-1, -1]
            continue

        # Get topic name and parent for hierarchical structure
        topic_name = topic_info[topic_info['Topic']==topic_id]['Name'].values[0]
        parent_name = name_map.get(topic_id, None)

        # Map to original index and policy area
        original_idx = index_mapping[doc_idx]
        policy_area = df.loc[original_idx]['policy_area']  # Use original policy_area from data
        
        # Organize in three-level structure
        doc_topic_levels[policy_area][parent_name][topic_name].append(original_idx)
    
    # Create taxonomy entries
    taxonomy = []
    
    # Build the taxonomy from the hierarchical structure
    for policy_area in doc_topic_levels:  # policy_area is already the original value
        for level1 in doc_topic_levels[policy_area]:
            for level2 in doc_topic_levels[policy_area][level1]:
                doc_indices = doc_topic_levels[policy_area][level1][level2]
                
                taxonomy.append({
                    'policy_area': policy_area,  # Use original policy_area
                    'subtopic_1': level1,        # Parent topic as first level
                    'subtopic_2': level2,        # Child topic as second level
                    'cluster_length': len(doc_indices),
                    'source_indices': ";".join(map(str, doc_indices))
                })
    
    # Convert to DataFrame and save
    taxonomy_df = pd.DataFrame(taxonomy)
    taxonomy_df.to_csv(output_file, index=False)
    print(f"Taxonomy saved to {output_file}")
    
    # Save the full topic model for future analysis
    topic_model.save("bertopic_model")
    print("BERTopic model saved for future reference")

if __name__ == "__main__":
    main()