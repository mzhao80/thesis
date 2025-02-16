import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from tqdm.auto import tqdm
import json
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("./topic_taxonomies_outputs", "keybert_debug.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./topic_taxonomies_outputs"

def setup_keybert():
    """Initialize KeyBERT with GPU support"""
    logger.info("Initializing KeyBERT with GPU support...")
    try:
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        logger.info("Successfully initialized SentenceTransformer on GPU")
        model = KeyBERT(model=st_model)
        logger.info("Successfully initialized KeyBERT")
        return model
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

def clean_text(text):
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

def process_keybert(df, kw_model, use_seed=False):
    """Extract keywords using KeyBERT"""
    results = []
    stats = {
        'total': len(df),
        'empty_text': 0,
        'extraction_errors': 0,
        'no_keywords': 0,
        'fallback_used': 0,
        'successful': 0
    }
    
    logger.info(f"Starting {'seeded ' if use_seed else ''}KeyBERT processing for {len(df)} documents")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), 
                      desc=f"Processing {'seeded ' if use_seed else ''}KeyBERT"):
        # Clean the text
        text = clean_text(row['speech'])
        if not text:
            stats['empty_text'] += 1
            logger.debug(f"Empty text for speech_id: {row['speech_id']}")
            results.append({
                'speech_id': row['speech_id'],
                'keybert_keywords': []
            })
            continue
            
        try:
            # Log text stats
            logger.debug(f"Processing text (id: {row['speech_id']}, length: {len(text.split())} words)")
            
            # Extract keywords
            if use_seed and not pd.isna(row['policy_area']):
                seed_words = [word.strip() for word in row['policy_area'].lower().split()]
                logger.debug(f"Using seed words: {seed_words}")
                keywords = kw_model.extract_keywords(
                    text,
                    seed_keywords=seed_words,
                    top_n=5,
                    stop_words='english',
                    use_maxsum=True,
                    diversity=0.5
                )
            else:
                keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    use_maxsum=True,
                    top_n=5,
                    diversity=0.5
                )
            
            # Log raw keywords and scores
            logger.debug(f"Raw keywords and scores: {keywords}")
            
            # Extract keywords with score threshold
            extracted = [kw for kw, score in keywords if score > 0.2]
            
            # Try fallback if no keywords found
            if not extracted:
                logger.debug(f"No keywords found with primary method, trying fallback for id: {row['speech_id']}")
                stats['fallback_used'] += 1
                
                keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_mmr=True,
                    top_n=3,
                    diversity=0.7
                )
                extracted = [kw for kw, score in keywords if score > 0.1]
                logger.debug(f"Fallback keywords and scores: {keywords}")
            
            if extracted:
                stats['successful'] += 1
                logger.debug(f"Successfully extracted keywords: {extracted}")
            else:
                stats['no_keywords'] += 1
                logger.warning(f"No keywords found for speech_id: {row['speech_id']}")
                logger.debug(f"Text preview: {text[:200]}...")
            
        except Exception as e:
            stats['extraction_errors'] += 1
            logger.error(f"Error processing text (id: {row['speech_id']}): {str(e)}")
            extracted = []
        
        results.append({
            'speech_id': row['speech_id'],
            'keybert_keywords': extracted
        })
    
    # Log final statistics
    logger.info("\nKeyBERT Processing Statistics:")
    logger.info(f"Total documents processed: {stats['total']}")
    logger.info(f"Empty texts skipped: {stats['empty_text']}")
    logger.info(f"Extraction errors: {stats['extraction_errors']}")
    logger.info(f"Documents with no keywords: {stats['no_keywords']}")
    logger.info(f"Fallback method used: {stats['fallback_used']}")
    logger.info(f"Successful extractions: {stats['successful']}")
    logger.info(f"Success rate: {(stats['successful'] / stats['total']) * 100:.2f}%")
    
    return pd.DataFrame(results)

def main():
    logger.info("Starting KeyBERT GPU processing")
    
    try:
        # Read the data
        logger.info("Reading input data...")
        df = pd.read_csv("df_bills.csv")
        logger.info(f"Loaded {len(df)} documents")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Initialize models
        logger.info("Initializing GPU models...")
        kw_model = setup_keybert()
        
        # Process with KeyBERT (no seed)
        logger.info("\nStarting regular KeyBERT extraction...")
        df_keybert = process_keybert(df, kw_model, use_seed=False)
        output_path = os.path.join(OUTPUT_DIR, "keybert_keywords.csv")
        df_keybert.to_csv(output_path, index=False)
        logger.info(f"Saved regular keywords to {output_path}")
        
        # Process with KeyBERT (with seed)
        logger.info("\nStarting seeded KeyBERT extraction...")
        df_keybert_seed = process_keybert(df, kw_model, use_seed=True)
        output_path = os.path.join(OUTPUT_DIR, "keybert_seed_keywords.csv")
        df_keybert_seed.to_csv(output_path, index=False)
        logger.info(f"Saved seeded keywords to {output_path}")
        
        logger.info("GPU processing complete!")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
