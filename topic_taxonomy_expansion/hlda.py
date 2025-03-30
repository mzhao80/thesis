#!/usr/bin/env python
"""
Hierarchical Latent Dirichlet Allocation (hLDA)

This script implements a hierarchical topic modeling approach using Hierarchical Latent Dirichlet
Allocation (hLDA) to create an alternative three-level taxonomy. Unlike the standard approach
which uses a combination of embedding-based clustering and LLM labeling, this approach directly
models the hierarchical structure of topics from the text data.

The script:
1. Loads and preprocesses the training data
2. Trains an hLDA model with specified parameters
3. Extracts the topic hierarchy and document assignments
4. Organizes documents into the three-level taxonomy
5. Outputs the taxonomy to a CSV file

Input:
- Training data CSV from the lab directory

Output:
- hlda_taxonomy.csv: Contains the hierarchical taxonomy with policy areas, subtopics, 
  and source document indices
- hlda_topic_words.csv: Contains the top words for each topic at each level
"""

import os
import pandas as pd
import numpy as np
import tomotopy as tp
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
import re

# Download NLTK resources (commented out as they only need to be run once)
#nltk.download('punkt')
#nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase and removing special characters.
    
    Args:
        text (str): The input text to preprocess
    
    Returns:
        str: The preprocessed text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """
    Remove common English stopwords from text.
    
    Args:
        text (str): The input text
    
    Returns:
        str: Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def main():
    """
    Main function to run the hLDA analysis and generate the taxonomy.
    """
    # Define paths
    input_file = '/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/training_data.csv'
    output_file = 'hlda_taxonomy.csv'
    
    # Parameters for hLDA
    depth = 3  # Three-level taxonomy
    
    print("Loading and preprocessing data...")
    # Load data
    df = pd.read_csv(input_file)
    df.fillna("", inplace=True)
    
    # Combine fields for better topic modeling
    df['combined_text'] = df['document'] + " " + df['summary']
    
    # Preprocess text
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    df['processed_text'] = df['processed_text'].apply(remove_stopwords)
    
    # Remove rows with empty processed text
    df = df[df['processed_text'].str.strip() != ""]
    df = df.reset_index(drop=True)
    
    # Create index mapping to keep track of original indices
    index_mapping = {i: df.index[i] for i in range(len(df))}
    
    # Prepare documents for hLDA
    docs = [doc.split() for doc in df['processed_text']]
    
    print("Training hLDA model...")
    # Initialize and train hLDA model
    seed = 42
    # The key parameters that control the number of topics in hLDA:
    # - gamma: controls the tendency to allocate documents to new topics
    #   Lower gamma values create more topics
    # - alpha: affects the document-topic distribution
    #   Lower alpha makes documents focus on fewer topics
    # - eta: controls word distribution in topics
    #   Lower eta creates more distinctive topics
    hlda = tp.HLDAModel(
        tw=tp.TermWeight.ONE,
        depth=depth, 
        alpha=0.15,  # Reduced from 0.1 to focus documents on fewer topics
        gamma=0.15,  # Reduced from 0.1 to encourage more topics at each level
        seed=seed
    )
    
    # Add documents to the model
    for doc in tqdm(docs, desc="Adding documents"):
        if doc:  # Skip empty documents
            hlda.add_doc(doc)
    
    # Train the model with iterations
    iterations = 200  # Increased from default to give model more time to converge
    for i in tqdm(range(iterations), desc="Training hLDA model"):
        hlda.train(1)
        current_ll = hlda.ll_per_word
        print(f"Iteration {i+1}/{iterations}, Log-likelihood per word: {current_ll:.4f}")
    
    # Print the number of topics at each level
    topic_counts = defaultdict(int)
    for doc in hlda.docs:
        path = list(doc.path)
        for level, topic_id in enumerate(path):
            topic_counts[(level, topic_id)] += 1
    
    level_counts = defaultdict(int)
    for (level, _) in topic_counts:
        level_counts[level] += 1
    
    print("\nNumber of topics discovered at each level:")
    for level in range(depth):
        print(f"Level {level}: {level_counts[level]} topics")
    
    print("Generating taxonomy from hLDA model...")
    # Organize document-topic assignments
    doc_topic_path = []
    
    for i, doc in enumerate(hlda.docs):
        path = list(doc.path)  # Get the path of topics for this document
        if len(path) == depth:  # Ensure we have a complete path
            doc_topic_path.append((i, path))
    
    # Organize into hierarchy
    taxonomy = []
    topic_docs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Group documents by their topic path and policy area
    for doc_idx, path in doc_topic_path:
        if len(path) >= 3:
            level1 = path[0]
            level2 = path[1]
            level3 = path[2]
            original_idx = index_mapping[doc_idx]
            policy_area = df.iloc[doc_idx]['policy_area']  # Use original policy_area from data
            topic_docs[policy_area][level2][level3].append(original_idx)
    
    # Extract top words for each topic at each level to use as labels
    topic_words = {}
    for level in range(depth):
        topic_words[level] = {}
        for k in range(hlda.k):
            if k < len(hlda.get_count_by_topics()):
                top_words = [word for word, _ in hlda.get_topic_words(k, top_n=10)]
                topic_words[level][k] = " ".join(top_words)
    
    # Create taxonomy entries
    for policy_area in topic_docs:
        for level1 in topic_docs[policy_area]:
            subtopic_1 = topic_words[0].get(level1, "Subtopic_" + str(level1))
            for level2 in topic_docs[policy_area][level1]:
                subtopic_2 = topic_words[1].get(level2, "Subtopic_" + str(level2))
                doc_indices = topic_docs[policy_area][level1][level2]
                
                taxonomy.append({
                    'policy_area': policy_area,  # Use the original policy area
                    'subtopic_1': subtopic_1,
                    'subtopic_2': subtopic_2,
                    'cluster_length': len(doc_indices),
                    'source_indices': ";".join(map(str, doc_indices))
                })
    
    # Convert to DataFrame and save
    taxonomy_df = pd.DataFrame(taxonomy)
    taxonomy_df.to_csv(output_file, index=False)
    print(f"Taxonomy saved to {output_file}")
    
    # Save topic words for reference
    topic_words_df = pd.DataFrame()
    for level in range(depth):
        level_words = {f"Topic_{k}": words for k, words in topic_words[level].items()}
        level_df = pd.DataFrame([level_words])
        level_df['level'] = level
        topic_words_df = pd.concat([topic_words_df, level_df], ignore_index=True)
    
    topic_words_df.to_csv('hlda_topic_words.csv', index=False)
    print("Topic words saved to hlda_topic_words.csv")

if __name__ == "__main__":
    main()