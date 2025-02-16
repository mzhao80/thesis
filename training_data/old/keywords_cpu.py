import os
import pandas as pd
import spacy
from tqdm.auto import tqdm
import json
import re
from openai import OpenAI

OUTPUT_DIR = "./topic_taxonomies_outputs"

def preprocess_text(text):
    """Preprocess text by removing speaker references and procedural phrases"""
    if pd.isna(text):
        return ""
        
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
    
    return text.strip()

def setup_spacy():
    """Initialize spaCy for CPU processing"""
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("sentencizer")
    return nlp

def process_spacy_batch(batch_data):
    """Process a batch of texts with spaCy"""
    nlp = setup_spacy()
    results = []
    
    texts = [preprocess_text(row['speech']) for row in batch_data]
    docs = nlp.pipe(texts)
    
    for doc, row in zip(docs, batch_data):
        noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks
                       if not chunk.text.isspace()]
        for ent in doc.ents:
            if ent.label_ in ["LAW", "ORG", "NORP"]:
                noun_phrases.append(ent.text)
        unique_noun_phrases = list(set(noun_phrases))
        results.append({
            'speech_id': row['speech_id'],
            'spacy_keywords': unique_noun_phrases
        })
    
    return results

def process_with_gpt(text, client):
    """Process a single text with GPT-4"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract up to 3 concise keywords or short phrases that summarize the main topics. Respond with just the keywords, separated by commas."},
                {"role": "user", "content": preprocess_text(text)}
            ],
            max_tokens=100,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in GPT processing: {str(e)}")
        return ""

def main():
    # Load the data
    df = pd.read_csv("df_bills.csv")
    
    # Process only first 500 speeches for GPT
    df_gpt = df.head(500).copy()
    print(f"Processing {len(df_gpt)} speeches with GPT...")
    
    # Initialize OpenAI client
    client = OpenAI(api_key="api_key")
    
    # Process speeches in batches
    batch_size = 5  # Process 5 speeches at a time
    results = []
    
    for i in tqdm(range(0, len(df_gpt), batch_size), desc="Processing batches"):
        batch = df_gpt.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            try:
                keywords = process_with_gpt(row['speech'], client)
                results.append({
                    'speech_id': row['speech_id'],
                    'gpt_keywords': keywords
                })
            except Exception as e:
                print(f"Error processing speech {row['speech_id']}: {str(e)}")
                continue
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "gpt_keywords.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Process with spaCy
    print("Starting spaCy keyword extraction...")
    batch_size = 100
    df_batches = [df[i:i + batch_size].to_dict('records') 
                  for i in range(0, len(df), batch_size)]
    
    spacy_keywords_list = []
    for batch_data in tqdm(df_batches, desc="Processing spaCy batches"):
        spacy_keywords_list.extend(process_spacy_batch(batch_data))
    
    df_spacy = pd.DataFrame(spacy_keywords_list)
    df_spacy.to_csv(os.path.join(OUTPUT_DIR, "spacy_keywords.csv"), index=False)
    
    print("Keyword extraction complete!")

if __name__ == "__main__":
    main()
