import os
import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./topic_taxonomies_outputs/master_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./topic_taxonomies_outputs"

def load_json_file(filepath):
    """Load a JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

def load_keyword_file(filepath, keyword_type=None):
    """Load a keyword file with error handling"""
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Handle list of dictionaries format
                if isinstance(data, list):
                    # Convert comma-separated strings to lists
                    return {item['speech_id']: [k.strip() for k in str(item.get('gpt_keywords', '')).split(',') if k.strip()] 
                            for item in data}
                return data
        
        df = pd.read_csv(filepath)
        # Handle different column name formats
        speech_id_col = next((col for col in df.columns if 'speech_id' in col.lower()), None)
        
        # Handle different keyword column names
        if keyword_type:
            keyword_col = next((col for col in df.columns if keyword_type.lower() in col.lower()), None)
        else:
            keyword_col = next((col for col in df.columns if 'keyword' in col.lower()), None)
        
        if not speech_id_col or not keyword_col:
            logger.warning(f"File {filepath} missing required columns. Found columns: {df.columns.tolist()}")
            return {}
            
        # Convert keywords string to list
        def parse_keywords(x):
            if pd.isna(x):
                return []
            if isinstance(x, str):
                try:
                    # Try to evaluate as a Python list
                    val = eval(x)
                    if isinstance(val, list):
                        # Filter out empty strings and strip quotes
                        return [k.strip("'\"") for k in val if k.strip()]
                    return [val] if val else []
                except:
                    # If eval fails, split by comma
                    return [k.strip() for k in x.split(',') if k.strip()]
            return x if isinstance(x, list) else [str(x)]
            
        df[keyword_col] = df[keyword_col].apply(parse_keywords)
        return dict(zip(df[speech_id_col].astype(str), df[keyword_col]))
            
    except Exception as e:
        logger.warning(f"Could not load {filepath}: {str(e)}")
        return {}

def extract_energy_hierarchy(hlda_data, bertopic_data):
    """Extract and format hierarchy information for energy policy area"""
    energy_info = {
        'hlda': None,
        'bertopic': None
    }
    
    # Extract hLDA hierarchy for energy
    if hlda_data and 'Energy' in hlda_data:
        energy_hlda = hlda_data['Energy']
        
        # Format hLDA hierarchy
        formatted_hlda = []
        if 'hierarchy' in energy_hlda and 'levels' in energy_hlda['hierarchy']:
            for level_idx, level in enumerate(energy_hlda['hierarchy']['levels']):
                level_topics = []
                for topic in level['topics']:
                    # Format topic words with scores
                    topic_words = [f"{w['word']}({w['score']:.3f})" for w in topic['words']]
                    level_topics.append({
                        'topic_id': topic['topic_id'],
                        'words': topic_words,
                        'n_docs': topic['n_docs']
                    })
                formatted_hlda.append({
                    'level': level_idx,
                    'topics': level_topics
                })
        energy_info['hlda'] = formatted_hlda
    
    # Extract BERTopic hierarchy for energy
    if bertopic_data and 'Energy' in bertopic_data:
        energy_bertopic = bertopic_data['Energy']
        
        # Format BERTopic hierarchy
        if 'hierarchy' in energy_bertopic:
            energy_info['bertopic'] = energy_bertopic['hierarchy']
    
    # Save energy hierarchies to separate file
    output_path = os.path.join(OUTPUT_DIR, "energy_hierarchies.json")
    with open(output_path, 'w') as f:
        json.dump(energy_info, f, indent=2)
    logger.info(f"Saved energy hierarchies to {output_path}")
    
    return energy_info

def parse_list_field(field):
    """Parse a field that might contain a list"""
    if pd.isna(field):
        return []
    if isinstance(field, str):
        try:
            val = eval(field)
            if isinstance(val, list):
                return val
            return [val] if val else []
        except:
            return [x.strip() for x in field.split(',') if x.strip()]
    return field if isinstance(field, list) else []

def format_committee(committee):
    """Format a committee entry preserving hierarchy"""
    if isinstance(committee, dict):
        result = committee['name']
        if committee.get('subcommittees'):
            subcommittees = [sub['name'] for sub in committee['subcommittees']]
            if subcommittees:
                result += f" ({', '.join(subcommittees)})"
        return result
    return str(committee)

def format_legislative_subject(subject):
    """Format a legislative subject entry"""
    if isinstance(subject, dict):
        return subject['name']
    return str(subject)

def main():
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load the main bills dataframe
        logger.info("Loading bills data...")
        df = pd.read_csv("df_bills.csv")

        # Filter out all speeches with less than 250 characters
        df = df[df['speech'].str.len() >= 250]
        
        # Load keyword results
        logger.info("Loading keyword results...")
        spacy_results = load_keyword_file(os.path.join(OUTPUT_DIR, "spacy_keywords.csv"), "spacy")
        keybert_results = load_keyword_file(os.path.join(OUTPUT_DIR, "keybert_keywords.csv"), "keybert")
        keybert_seeded_results = load_keyword_file(os.path.join(OUTPUT_DIR, "keybert_seed_keywords.csv"), "keybert")
        gpt_results = load_keyword_file(os.path.join(OUTPUT_DIR, "gpt_keywords.json"), "gpt")
        
        # Load LDA results
        logger.info("Loading LDA results...")
        try:
            with open(os.path.join(OUTPUT_DIR, "lda_keywords.json"), 'r') as f:
                lda_results = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load LDA results: {str(e)}")
            lda_results = {}
        
        # Load BERTopic results
        logger.info("Loading BERTopic results...")
        try:
            with open(os.path.join(OUTPUT_DIR, "bertopic_keywords.json"), 'r') as f:
                bertopic_results = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load BERTopic results: {str(e)}")
            bertopic_results = {}
        
        # Load hierarchy results
        logger.info("Loading hierarchy results...")
        hlda_results = load_json_file(os.path.join(OUTPUT_DIR, "hlda_hierarchy.json"))
        bertopic_hierarchy_results = load_json_file(os.path.join(OUTPUT_DIR, "bertopic_hierarchy.json"))
        
        # Extract energy policy area hierarchies
        logger.info("Processing energy policy area hierarchies...")
        energy_hierarchies = extract_energy_hierarchy(hlda_results, bertopic_hierarchy_results)
        
        # Create combined dataframe
        logger.info("Combining results...")
        combined_data = []
        
        for _, row in df.iterrows():
            speech_id = row['speech_id']
            
            # Parse and format lists
            legislative_subjects = parse_list_field(row.get('legislative_subjects', []))
            legislative_subjects = [format_legislative_subject(s) for s in legislative_subjects]
            
            committees = parse_list_field(row.get('committees', []))
            committees = [format_committee(c) for c in committees]
            
            entry = {
                'speech_id': speech_id,
                'speech': row['speech'],
                'standardized_title': row['standardized_title'],
                'policy_area': row['policy_area'],
                'legislative_subjects': '|'.join(legislative_subjects),
                'committees': '|'.join(committees),
                'spacy_keywords': '|'.join(spacy_results.get(speech_id, [])),
                'keybert_keywords': '|'.join(keybert_results.get(speech_id, [])),
                'keybert_seeded_keywords': '|'.join(keybert_seeded_results.get(speech_id, [])),
                'lda_keywords': '|'.join(lda_results.get(speech_id, [])),
                'bertopic_keywords': '|'.join(bertopic_results.get(speech_id, [])),
                'gpt_keywords': '|'.join(gpt_results.get(speech_id, []))
            }
            combined_data.append(entry)
        
        # Convert to dataframe and save
        combined_df = pd.DataFrame(combined_data)
        
        # Save as CSV
        csv_path = os.path.join(OUTPUT_DIR, "combined_results.csv")
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved combined results to {csv_path}")
        
        # Save as JSON for better preservation of nested structures
        json_path = os.path.join(OUTPUT_DIR, "combined_results.json")
        with open(json_path, 'w') as f:
            json.dump(combined_data, f, indent=2)
        logger.info(f"Saved combined results to {json_path}")
        
        # Generate summary statistics
        logger.info("\nGenerating summary statistics...")
        
        # Document counts
        logger.info(f"Total documents: {len(combined_df)}")
        logger.info(f"Documents by policy area:\n{combined_df['policy_area'].value_counts()}")
        
        # Keyword statistics
        def count_items(x):
            return len([i for i in x.split('|') if i.strip()])
            
        keyword_stats = {
            'spacy': combined_df['spacy_keywords'].apply(count_items).mean(),
            'keybert': combined_df['keybert_keywords'].apply(count_items).mean(),
            'keybert_seeded': combined_df['keybert_seeded_keywords'].apply(count_items).mean(),
            'gpt': combined_df['gpt_keywords'].apply(count_items).mean(),
            'lda': combined_df['lda_keywords'].apply(count_items).mean(),
            'bertopic': combined_df['bertopic_keywords'].apply(count_items).mean()
        }
        logger.info("\nAverage keywords per document:")
        for method, avg in keyword_stats.items():
            logger.info(f"  {method}: {avg:.2f}")
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error("Fatal error in main", exc_info=True)
        raise

if __name__ == "__main__":
    main()
