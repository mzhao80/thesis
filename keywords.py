import os
import pandas as pd
import tomotopy as tp
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from openai import OpenAI
import numpy as np
import json
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# Get number of CPUs for parallel processing
NUM_CPUS = multiprocessing.cpu_count()

###############################################################################
# 1. Read the CSV data
###############################################################################
df = pd.read_csv("df_bills.csv")

# Keep additional columns
columns_to_keep = ['speech', 'standardized_title', 'policy_area', 'legislative_subjects', 'committees', 'speech_id']
df = df[columns_to_keep]

###############################################################################
# 2. spaCy Setup and Text Preprocessing
###############################################################################
try:
    spacy.require_gpu()
    print("spaCy is set to use GPU.")
except:
    print("spaCy GPU not available or not installed with GPU support. Using CPU.")

nlp = spacy.load("en_core_web_sm")
# Enable parallel processing for spaCy
# get maximum speech length from df_bills['speech']
nlp.max_length = df['speech'].str.len().max()
if not spacy.prefer_gpu():
    nlp.add_pipe("sentencizer")  # Add sentencizer for CPU processing

def preprocess_text(doc_text):
    """
    Basic preprocessing using spaCy:
    1. Convert to lowercase
    2. Remove stop words, punctuation, whitespace
    3. Return list of lemmatized tokens
    """
    doc = nlp(doc_text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not (token.is_stop or token.is_punct or token.is_space)
    ]
    return tokens

def process_texts_batch(texts):
    """Process a batch of texts using spaCy's pipe for efficiency"""
    docs = nlp.pipe(texts)
    return [preprocess_text(doc.text) for doc in docs]

###############################################################################
# 3. Policy Area-specific hLDA Analysis
###############################################################################
# def run_hlda_for_group(texts, depth=3):
#     """Run hLDA for a group of texts and return the model and results"""
#     corpus = [preprocess_text(text) for text in texts]
    
#     model = tp.HLDAModel(
#         tw=tp.TermWeight.PMI,
#         min_cf=3,
#         depth=depth,
#         alpha=0.1,
#         gamma=1,
#         eta=0.01
#     )
    
#     for tokens in corpus:
#         model.add_doc(tokens)
    
#     # Train hLDA
#     model.burn_in = 100
#     model.train(1000)
    
#     # Collect results
#     results = []
#     for i, doc in enumerate(model.docs):
#         topic_path = doc.path
#         path_topics_words = []
#         for level, topic_id in enumerate(topic_path):
#             top_words = model.get_topic_words(topic_id, top_n=5)
#             top_words_str = ", ".join([w for w, _ in top_words])
#             path_topics_words.append({
#                 "level": level,
#                 "topic_id": topic_id,
#                 "words": top_words_str
#             })
#         results.append({
#             "doc_index": i,
#             "topic_path": path_topics_words
#         })
    
#     return model, results

# # Group by policy area and run hLDA for each group
# policy_area_results = {}
# for policy_area in tqdm(df['policy_area'].unique(), desc="Processing policy areas"):
#     if pd.isna(policy_area):
#         continue
        
#     print(f"\nProcessing policy area: {policy_area}")
#     area_texts = df[df['policy_area'] == policy_area]['speech'].tolist()
    
#     if len(area_texts) < 5:  # Skip areas with too few documents
#         print(f"Skipping {policy_area} - too few documents ({len(area_texts)})")
#         continue
        
#     _, results = run_hlda_for_group(area_texts)
    
#     policy_area_results[policy_area] = {
#         "num_documents": len(area_texts),
#         "topic_hierarchy": results
#     }

# # Save policy area-specific results
# output_dir = "./topic_taxonomies_outputs"
# os.makedirs(output_dir, exist_ok=True)
# with open(os.path.join(output_dir, "policy_area_topics.json"), "w") as f:
#     json.dump(policy_area_results, f, indent=2)


###############################################################################
# 4. Hierarchical LDA (hLDA) on the Entire Corpus
###############################################################################
print("Preprocessing texts for hLDA...")
# Process texts in parallel batches
batch_size = 1000
text_batches = [df['speech'][i:i + batch_size].tolist() 
                for i in range(0, len(df), batch_size)]

with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
    corpus_hlda = []
    for batch_tokens in tqdm(
        executor.map(process_texts_batch, text_batches),
        total=len(text_batches),
        desc="Processing text batches"
    ):
        corpus_hlda.extend(batch_tokens)

hlda_model = tp.HLDAModel(
    tw=tp.TermWeight.PMI,
    min_cf=3,  # Minimum corpus frequency for a term
    depth=3,   # Maximum depth of the hLDA tree
    alpha=0.1,
    gamma=1,
    eta=0.01
)

for tokens in corpus_hlda:
    hlda_model.add_doc(tokens)

# Train hLDA
hlda_model.burn_in = 100
hlda_model.train(1000)

# Collect hLDA results
hlda_results = []
for i, doc in enumerate(hlda_model.docs):
    topic_path = doc.path  # e.g., [0, 1, 5]
    path_topics_words = []
    for level, topic_id in enumerate(topic_path):
        top_words = hlda_model.get_topic_words(topic_id, top_n=5)
        top_words_str = ", ".join([w for w, _ in top_words])
        path_topics_words.append(f"Level {level}: Topic {topic_id} => {top_words_str}")

    hlda_results.append({
        'speech_id': df.iloc[i]['speech_id'],
        'topic_words_path': " | ".join(path_topics_words)
    })

df_hlda = pd.DataFrame(hlda_results)

###############################################################################
# 5. Standard LDA (flat LDA) for comparison
###############################################################################
# Similar approach, but we specify a fixed number of topics (k)
corpus_lda = corpus_hlda

lda_k = 3
lda_model = tp.LDAModel(
    k=lda_k,
    alpha=0.1,
    eta=0.01,
    min_cf=3,
    tw=tp.TermWeight.PMI
)

for tokens in corpus_lda:
    lda_model.add_doc(tokens)

# Train LDA
lda_model.train(1000)

# Extract top words per topic for reference
# (You could store them for manual inspection)
for topic_id in range(lda_k):
    top_words = lda_model.get_topic_words(topic_id=topic_id, top_n=5)
    top_words_str = ", ".join([w for w, _ in top_words])
    print(f"Topic {topic_id}: {top_words_str}")

# Document-topic distributions
lda_results = []
for i, doc in enumerate(lda_model.docs):
    topic_dist = doc.get_topic_dist()  # e.g., [0.05, 0.7, 0.05, ...]
    top_topic = int(np.argmax(topic_dist))
    lda_results.append({
        "speech_id": df.iloc[i]["speech_id"],
        "top_lda_topic": top_topic,
        "topic_distribution": topic_dist
    })

df_lda = pd.DataFrame(lda_results)

###############################################################################
# 6. Keyword Extraction with spaCy (noun chunks)
###############################################################################
print("Extracting spaCy keywords...")
spacy_keywords_list = []

def process_spacy_batch(batch):
    results = []
    docs = nlp.pipe([row['speech'] for row in batch])
    for doc, row in zip(docs, batch):
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

# Process in batches
batch_size = 100
df_batches = [df[i:i + batch_size].to_dict('records') 
              for i in range(0, len(df), batch_size)]

with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
    for batch_results in tqdm(
        executor.map(process_spacy_batch, df_batches),
        total=len(df_batches),
        desc="Processing spaCy batches"
    ):
        spacy_keywords_list.extend(batch_results)

df_spacy = pd.DataFrame(spacy_keywords_list)

###############################################################################
# 7. KeyBERT
###############################################################################
print("Extracting KeyBERT keywords...")
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")  # or "cpu"
kw_model = KeyBERT(model=st_model)

def process_keybert_batch(batch):
    results = []
    for row in batch:
        keywords = kw_model.extract_keywords(
            row['speech'],
            keyphrase_ngram_range=(1, 5),
            stop_words='english',
            top_n=3,
            use_maxsum=True,
            nr_candidates=10,
        )
        extracted = [kw for kw, score in keywords]
        results.append({
            'speech_id': row['speech_id'],
            'keybert_keywords': extracted
        })
    return results

# Process in batches
with ThreadPoolExecutor(max_workers=NUM_CPUS) as executor:
    keybert_list = []
    for batch_results in tqdm(
        executor.map(process_keybert_batch, df_batches),
        total=len(df_batches),
        desc="Processing KeyBERT batches"
    ):
        keybert_list.extend(batch_results)

df_keybert = pd.DataFrame(keybert_list)

###############################################################################
# 7.5 KeyBERT (with seeding)
###############################################################################
print("Extracting seeded KeyBERT keywords...")
def process_keybert_seed_batch(batch):
    results = []
    for row in batch:
        keywords = kw_model.extract_keywords(
            row['speech'],
            keyphrase_ngram_range=(1, 5),
            stop_words='english',
            top_n=3,
            use_maxsum=True,
            nr_candidates=10,
            seed_keywords=row['policy_area']
        )
        extracted = [kw for kw, score in keywords]
        results.append({
            'speech_id': row['speech_id'],
            'keybert_keywords': extracted
        })
    return results

# Process in batches
with ThreadPoolExecutor(max_workers=NUM_CPUS) as executor:
    keybert_seed_list = []
    for batch_results in tqdm(
        executor.map(process_keybert_seed_batch, df_batches),
        total=len(df_batches),
        desc="Processing seeded KeyBERT batches"
    ):
        keybert_seed_list.extend(batch_results)

df_keybert_seed = pd.DataFrame(keybert_seed_list)

###############################################################################
# 8. GPT API for Keyword Extraction
###############################################################################
print("Extracting GPT keywords...")
client = OpenAI(api_key="api_key")

def process_gpt_batch(batch):
    results = []
    for row in batch:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract up to 3 concise keywords or short phrases that summarize the main topics. Respond with just the keywords, separated by commas."},
                    {"role": "user", "content": row['speech']}
                ],
                max_tokens=100,
                temperature=0.3,
                n=1
            )
            gpt_output = response.choices[0].message.content.strip()
            extracted_keywords = [k.strip(" ,;.-") for k in gpt_output.split(",") if k.strip()]
            results.append({
                'speech_id': row['speech_id'],
                'gpt_keywords': extracted_keywords
            })
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            results.append({
                'speech_id': row['speech_id'],
                'gpt_keywords': []
            })
    return results

# Process in smaller batches for API
gpt_batch_size = 10
gpt_batches = [df[i:i + gpt_batch_size].to_dict('records') 
               for i in range(0, len(df), gpt_batch_size)]

with ThreadPoolExecutor(max_workers=4) as executor:  # Limit concurrent API calls
    gpt_list = []
    for batch_results in tqdm(
        executor.map(process_gpt_batch, gpt_batches),
        total=len(gpt_batches),
        desc="Processing GPT batches"
    ):
        gpt_list.extend(batch_results)

df_gpt = pd.DataFrame(gpt_list)

###############################################################################
# 9. Merge All Results Into One CSV
###############################################################################
# Our final DataFrame will contain columns like:
#  [speech, spacy, hlda, lda, keybert, gpt]
df_final = df[['speech', 'standardized_title', 'policy_area', 'legislative_subjects', 'committees', 'speech_id']]

# spaCy
df_final = df_final.merge(df_spacy[['speech_id', 'spacy_keywords']], 
                          on='speech_id', how='left')
df_final.rename(columns={'spacy_keywords': 'spacy'}, inplace=True)

# Standard LDA
df_final = df_final.merge(df_lda[['speech_id', 'top_lda_topic']],
                          on='speech_id', how='left')
df_final.rename(columns={'top_lda_topic': 'lda'}, inplace=True)

# hLDA
df_final = df_final.merge(df_hlda[['speech_id', 'topic_words_path']],
                          on='speech_id', how='left')
df_final.rename(columns={'topic_words_path': 'hlda'}, inplace=True)

# KeyBERT
df_final = df_final.merge(df_keybert[['speech_id', 'keybert_keywords']],
                          on='speech_id', how='left')
df_final.rename(columns={'keybert_keywords': 'keybert'}, inplace=True)

#KeyBert Seed
df_final = df_final.merge(df_keybert_seed[['speech_id', 'keybert_keywords']],
                          on='speech_id', how='left')
df_final.rename(columns={'keybert_keywords': 'keybert_seed'}, inplace=True)

# GPT
df_final = df_final.merge(df_gpt[['speech_id', 'gpt_keywords']],
                          on='speech_id', how='left')
df_final.rename(columns={'gpt_keywords': 'gpt'}, inplace=True)

# Final column ordering
df_final = df_final[['speech', 'standardized_title', 'policy_area', 'legislative_subjects', 'committees', 'spacy', 'hlda', 'lda', 'keybert', 'keybert_seed', 'gpt']]

# Save to a single CSV
output_dir = "./topic_taxonomies_outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "combined_topic_keywords.csv")
df_final.to_csv(output_path, index=False)

print("Done! Results are in:", output_path)
