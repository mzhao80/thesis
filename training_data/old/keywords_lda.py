import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
import os
import logging
from datetime import datetime
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./topic_taxonomies_outputs"

def preprocess_text(text):
    """Clean text for keyword extraction"""
    if pd.isna(text):
        return ""
    text = text.strip()
        # Replace Madam Speaker, Mr. President, Madam President with Mr. Speaker
    text = re.sub(r'Mr\. President', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chair', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Speakerman', 'Mr. Speaker', text)
    text = re.sub(r'Madam President', 'Mr. Speaker', text)
    text = re.sub(r'Madam Speaker', 'Mr. Speaker', text)
    text = re.sub(r'Madam Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chair', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairwoman', 'Mr. Speaker', text)

    # strip out the following phrases from the beginning of each text and leave the remainder:
    # "Mr. Speaker, " 
    text = re.sub(r'^Mr\. Speaker, ', '', text)
    # "Mr. Speaker, I yield myself the balance of my time. "
    text = re.sub(r'^I yield myself the balance of my time\. ', '', text)
    # "I yield myself such time as I may consume. "
    text = re.sub(r'^I yield myself such time as I may consume\. ', '', text)
    

    return text

def extract_topics_lda(texts, n_topics=10, n_words=2):
    """
    Extract topics using LDA
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        n_words: Number of words per topic to return
    Returns:
        topics: List of topics, where each topic is a list of words
        doc_topics: Topic distribution for each document
        vectorizer: The fitted CountVectorizer
        model: The fitted LDA model
    """
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.9,  # Remove very common words
        min_df=2,     # Remove very rare words
        stop_words='english',
        max_features=10000
    )
    
    logger.info("Creating document-term matrix...")
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Train LDA model
    logger.info("Training LDA model...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch'
    )
    
    # Get document-topic distribution
    doc_topics = lda_model.fit_transform(doc_term_matrix)
    
    # Get topic-word distribution
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
    
    return topics, doc_topics, vectorizer, lda_model

def get_document_topics(doc_topics, topics, threshold=0.1):
    """
    Get the main topics for each document
    Args:
        doc_topics: Topic distribution for each document
        topics: List of topics (each topic is a list of words)
        threshold: Minimum probability to consider a topic relevant
    Returns:
        List of lists, where each inner list contains the relevant topics for a document
    """
    doc_main_topics = []
    for doc_dist in doc_topics:
        # Get topics above threshold
        relevant_topics = []
        for topic_idx, prob in enumerate(doc_dist):
            if prob >= threshold:
                relevant_topics.append({
                    'topic_id': topic_idx,
                    'probability': float(prob),
                    'words': topics[topic_idx]
                })
        # Sort by probability
        relevant_topics.sort(key=lambda x: x['probability'], reverse=True)
        doc_main_topics.append(relevant_topics)
    return doc_main_topics

def main():
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv("df_bills.csv")
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        texts = [preprocess_text(text) for text in df['speech']]
        
        # Extract topics
        logger.info("Extracting topics...")
        topics, doc_topics, vectorizer, model = extract_topics_lda(texts)
        
        # Get document topics
        logger.info("Getting document topics...")
        doc_main_topics = get_document_topics(doc_topics, topics)
        
        # Create results dictionary
        results = {}
        for idx, (speech_id, topics) in enumerate(zip(df['speech_id'], doc_main_topics)):
            # Format topics as a list of words with probabilities
            topic_words = []
            for topic in topics:
                # Combine words with probabilities
                topic_str = f"{' '.join(topic['words'])} ({topic['probability']:.3f})"
                topic_words.append(topic_str)
            results[speech_id] = topic_words
        
        # Save results
        output_file = os.path.join(OUTPUT_DIR, "lda_keywords.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_file}")
        
        # Save model info
        model_info = {
            'n_topics': model.n_components,
            'vocab_size': len(vectorizer.get_feature_names_out()),
            'perplexity': model.perplexity(vectorizer.transform(texts))
        }
        info_file = os.path.join(OUTPUT_DIR, "lda_model_info.json")
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Saved model info to {info_file}")
        
    except Exception as e:
        logger.error("Error in main", exc_info=True)
        raise

if __name__ == "__main__":
    main()
