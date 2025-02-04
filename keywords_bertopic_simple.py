import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import json
import os
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./topic_taxonomies_outputs"

def preprocess_text(text):
    """Preprocess text by removing speaker references and procedural phrases"""
    if pd.isna(text):
        return ""
        
    # Replace various speaker references with Mr. Speaker
    speaker_patterns = [
        r'Mr\. President', r'Mr\. Clerk', r'Mr\. Chair', r'Mr\. Chairman',
        r'Mr\. Speakerman', r'Madam President', r'Madam Speaker', r'Madam Clerk',
        r'Madam Chair', r'Madam Chairman', r'Madam Chairwoman'
    ]
    for pattern in speaker_patterns:
        text = re.sub(pattern, 'Mr. Speaker', text)

    # Remove common procedural phrases
    procedural_patterns = [
        r'^Mr\. Speaker, ',
        r'^I yield myself the balance of my time\. ',
        r'^I yield myself such time as I may consume\. '
    ]
    for pattern in procedural_patterns:
        text = re.sub(pattern, '', text)
    
    return text.strip()

def create_bertopic_model():
    """Create and configure BERTopic model"""
    # Initialize sentence transformer
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    
    # Configure UMAP
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # Configure HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=sentence_transformer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True
    )
    
    return topic_model

def extract_document_topics(topic_model, doc_topics, doc_probs, top_n=5):
    """Extract top topics for each document with their terms and probabilities"""
    results = []
    
    for topic, probs in zip(doc_topics, doc_probs):
        # Get indices of top N topics by probability
        top_indices = np.argsort(probs)[-top_n:][::-1]
        
        # Get topic terms and probabilities for each top topic
        doc_result = []
        for idx in top_indices:
            if probs[idx] > 0.05:  # Only include topics with >5% probability
                # Get the topic terms for the topic index
                topic_terms = topic_model.get_topic(idx)
                if topic_terms:  # Some topics might be -1 (no topic)
                    # Format as "term1 term2 term3 (probability)"
                    terms = " ".join([term for term, _ in topic_terms[:3]])
                    doc_result.append(f"{terms} ({probs[idx]:.3f})")
        
        results.append(doc_result)
    
    return results

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
        
        # Create and fit BERTopic model
        logger.info("Creating and fitting BERTopic model...")
        topic_model = create_bertopic_model()
        topics, probs = topic_model.fit_transform(texts)
        
        # Extract document topics
        logger.info("Extracting document topics...")
        doc_topics = extract_document_topics(topic_model, topics, probs)
        
        # Create results dictionary
        results = {}
        for speech_id, topics in zip(df['speech_id'], doc_topics):
            results[speech_id] = topics
        
        # Save results
        output_file = os.path.join(OUTPUT_DIR, "bertopic_keywords.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_file}")
        
        # Save model info
        model_info = {
            'num_topics': len(topic_model.get_topics()),
            'model_type': 'BERTopic',
            'embedding_model': 'all-MiniLM-L6-v2'
        }
        info_file = os.path.join(OUTPUT_DIR, "bertopic_model_info.json")
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Saved model info to {info_file}")
        
    except Exception as e:
        logger.error("Error in main", exc_info=True)
        raise

if __name__ == "__main__":
    main()
