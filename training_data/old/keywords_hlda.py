import os
import pandas as pd
import numpy as np
import tomotopy as tp
from tqdm.auto import tqdm
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./topic_taxonomies_outputs/hlda_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./topic_taxonomies_outputs"

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def run_hlda(texts, depth=3, iterations=100):
    """Run hierarchical LDA on a collection of texts"""
    if len(texts) < 5:
        logger.warning(f"Not enough documents ({len(texts)}) to run hLDA")
        return None
        
    logger.info(f"Training model with {len(texts)} documents...")
    
    try:
        # Calculate appropriate model parameters based on corpus size
        min_cf = max(2, int(len(texts) * 0.001))  # Minimum term frequency
        min_df = max(2, int(len(texts) * 0.01))   # Minimum document frequency
        rm_top = min(10, int(len(texts) * 0.05))  # Remove top frequent terms
        
        # Initialize model with adjusted parameters
        model = tp.HLDAModel(
            depth=depth,
            min_cf=min_cf,      # Minimum collection frequency for words
            rm_top=rm_top,      # Remove top frequent words
            tw=tp.TermWeight.ONE,
            gamma=1.0,         # Topic smoothing
            alpha=0.1,         # Document-topic smoothing
            eta=0.01,          # Topic-word smoothing
            seed=42            # For reproducibility
        )
        
        # Add documents with validation
        n_valid_docs = 0
        n_short_docs = 0
        n_empty_docs = 0
        
        for doc in texts:
            if pd.isna(doc):
                n_empty_docs += 1
                continue
                
            # Tokenize and clean
            tokens = str(doc).split()
            if len(tokens) < 2:
                n_short_docs += 1
                continue
                
            try:
                model.add_doc(tokens)
                n_valid_docs += 1
            except Exception as e:
                logger.warning(f"Error adding document: {str(e)}")
        
        logger.info(f"Document statistics:")
        logger.info(f"  Valid documents: {n_valid_docs}")
        logger.info(f"  Short documents (< 2 tokens): {n_short_docs}")
        logger.info(f"  Empty documents: {n_empty_docs}")
        
        if len(model.docs) < 5:
            logger.warning(f"Not enough valid documents ({len(model.docs)}) after filtering")
            return None
        
        # Train model with monitoring
        logger.info("Training hLDA model...")
        prev_ll = float('-inf')
        n_worse = 0
        best_ll = float('-inf')
        
        for i in range(iterations):
            model.train(1)
            current_ll = model.ll_per_word
            
            if (i + 1) % 10 == 0:
                logger.info(f"Iteration: {i+1}\tLog-likelihood: {current_ll}")
                
                # Check for convergence or problems
                if current_ll == float('-inf'):
                    logger.error("Model diverged (log-likelihood is -inf)")
                    return None
                    
                if current_ll < prev_ll:
                    n_worse += 1
                else:
                    n_worse = 0
                    
                if current_ll > best_ll:
                    best_ll = current_ll
                
                # Stop if likelihood getting worse
                if n_worse >= 3:
                    logger.warning(f"Stopping early at iteration {i+1} due to decreasing likelihood")
                    break
                    
                prev_ll = current_ll
        
        # Final validation
        if model.ll_per_word == float('-inf'):
            logger.error("Final model has -inf log-likelihood")
            return None
            
        if abs(model.ll_per_word - best_ll) > 1.0:
            logger.warning("Final model significantly worse than best model")
        
        # Extract hierarchical topic structure
        hierarchy = {
            'depth': depth,
            'n_docs': len(model.docs),
            'log_likelihood': float(model.ll_per_word),
            'parameters': {
                'min_cf': min_cf,
                'rm_top': rm_top
            },
            'levels': []
        }
        
        # Process each level
        for level in range(depth):
            level_topics = []
            
            # Get topics at this level using get_count_by_topics
            topic_counts = {}
            for doc_idx in range(len(model.docs)):
                doc_path = model.docs[doc_idx].path
                if len(doc_path) > level:
                    topic = doc_path[level]
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Process each topic at this level
            for topic, count in topic_counts.items():
                # Get top words for this topic
                words_and_scores = []
                for word, score in model.get_topic_words(topic):
                    if score > 0.001:  # Filter out very low probability words
                        words_and_scores.append({
                            'word': str(word),
                            'score': float(score)
                        })
                
                # Only include topics with meaningful words
                if words_and_scores:
                    # Get document assignments
                    doc_indices = []
                    for doc_idx in range(len(model.docs)):
                        doc_path = model.docs[doc_idx].path
                        if len(doc_path) > level and doc_path[level] == topic:
                            doc_indices.append(int(doc_idx))
                    
                    level_topics.append({
                        'topic_id': int(topic),
                        'words': words_and_scores,
                        'n_docs': len(doc_indices),
                        'doc_indices': doc_indices,
                        'topic_count': int(count)
                    })
            
            hierarchy['levels'].append({
                'level': level,
                'topics': level_topics
            })
        
        return hierarchy
        
    except Exception as e:
        logger.error(f"Error in run_hlda: {str(e)}", exc_info=True)
        return None

def process_policy_area(policy_area, texts):
    """Process a single policy area with hierarchical LDA"""
    logger.info(f"\nProcessing policy area: {policy_area}")
    logger.info(f"Number of documents: {len(texts)}")
    
    if len(texts) < 5:
        logger.warning(f"Skipping {policy_area}: too few documents")
        return None
    
    try:
        hierarchy = run_hlda(texts)
        if hierarchy is None:
            return None
            
        return {
            'policy_area': str(policy_area),
            'num_documents': len(texts),
            'hierarchy': hierarchy
        }
    except Exception as e:
        logger.error(f"Error processing {policy_area}: {str(e)}", exc_info=True)
        return None

def main():
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Read the data
        logger.info("Reading data...")
        df = pd.read_csv("df_bills.csv")
        logger.info(f"Loaded {len(df)} documents")
        
        # Process each policy area
        logger.info("\nProcessing policy areas...")
        policy_area_results = {}
        
        for policy_area in df['policy_area'].unique():
            if pd.isna(policy_area):
                continue
            
            area_texts = df[df['policy_area'] == policy_area]['speech'].tolist()
            results = process_policy_area(policy_area, area_texts)
            
            if results:
                policy_area_results[str(policy_area)] = convert_to_serializable(results)
        
        # Save policy area results
        logger.info("\nSaving policy area results...")
        output_path = os.path.join(OUTPUT_DIR, "hlda_hierarchy.json")
        with open(output_path, "w") as f:
            json.dump(policy_area_results, f, indent=2, cls=NumpyJSONEncoder)
        logger.info(f"Results saved to {output_path}")
        
        # Process global hierarchy if needed
        if len(df) > 0:
            logger.info("\nProcessing global hierarchy...")
            all_texts = df['speech'].tolist()
            global_hierarchy = run_hlda(all_texts, depth=4)
            
            if global_hierarchy:
                global_output = {
                    'num_documents': len(all_texts),
                    'hierarchy': convert_to_serializable(global_hierarchy)
                }
                
                output_path = os.path.join(OUTPUT_DIR, "hlda_global_hierarchy.json")
                with open(output_path, "w") as f:
                    json.dump(global_output, f, indent=2, cls=NumpyJSONEncoder)
                logger.info(f"Global hierarchy saved to {output_path}")
        
        logger.info("hLDA analysis complete!")
        
    except Exception as e:
        logger.error("Fatal error in main", exc_info=True)
        raise

if __name__ == "__main__":
    main()
