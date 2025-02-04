#!/usr/bin/env python
import os
import argparse
import pandas as pd
from tqdm import tqdm
import openai

def extract_topic_and_phrases(speech, policy, client, max_attempts=10):
    """
    For a given speech and its broad policy area, call the LLM to extract:
      - a short subtopic (1–5 words) describing the main issue in the speech (this will be considered a subtopic)
      - key phrases (separated by " | ") exactly quoted from the speech.
    Retries up to max_attempts until successful.
    Returns (subtopic, phrase_list)
    """
    prompt = (
        "Here we have the text of a congressional speech and a broad policy area assigned by the Congressional Research Service. For the following speech, please provide on two separate lines:\n"
        "1. On the first line, a short topic (1–5 words) that describes the main political issue discussed in the speech, followed by a newline.\n"
        "2. On the second line, a set of 1-3 key phrases (each 2–10 words) exactly quoted from the speech that best represent that topic, separated by ' | ', without any quotation marks.\n\n"
        f"Speech: {speech}\n\n"
        f"Broad Policy Area: {policy}\n\n"
        "Response (exactly two lines, with the first line being the topic followed by a newline, and the second line the key phrases separated by ' | '):\n"
    )
    messages = [
        {"role": "system",
         "content": (
             "You are a helpful assistant that extracts topics and key phrases from congressional speeches. "
             "Respond with exactly two lines: the first line is a short topic (1–5 words) followed by a newline, "
             "and the second line is the set of 1-3 key phrases quoted exactly from the text, separated by ' | ', with no quotation marks."
         )},
        {"role": "user", "content": prompt}
    ]
    
    attempt = 0
    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            content = response.choices[0].message.content.strip()
            lines = content.split('\n')
            if len(lines) < 2:
                raise ValueError("Expected at least two lines in response.")
            subtopic = lines[0].strip()
            phrases_line = lines[1].strip()
            phrase_list = [p.strip() for p in phrases_line.split('|') if p.strip()]
            return subtopic, phrase_list
        except Exception as e:
            print(f"Error parsing LLM response on attempt {attempt+1}: {e}\nResponse was: {content if 'content' in locals() else 'N/A'}")
            attempt += 1
    return "", []

def main():
    parser = argparse.ArgumentParser(description="Preprocess df_bills.csv to produce training_data.csv")
    parser.add_argument('--input-file', type=str, default='df_bills.csv',
                        help="Input CSV file (must contain 'speech' and 'policy_area' columns)")
    parser.add_argument('--output-file', type=str, default='training_data.csv',
                        help="Output CSV file with document, policy_area, subtopic, and phrase columns")
    parser.add_argument('--data-dir', type=str, default='topicexpan',
                        help="Directory to store output files")
    parser.add_argument('--batch-size', type=int, default=5,
                        help="Batch size for LLM calls (default: 5)")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    output_path = os.path.join(args.data_dir, args.output_file)
    
    # If the training CSV already exists, skip prompting.
    if os.path.exists(output_path):
        print("Training data file already exists; skipping LLM prompting.")
        out_df = pd.read_csv(output_path)
    else:
        print("Loading input CSV...")
        df = pd.read_csv(args.input_file)
        
        # Write corpus.txt: each line "doc_index<TAB>speech"
        corpus_path = os.path.join(args.data_dir, "corpus.txt")
        print("Writing corpus.txt ...")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for idx, row in df.iterrows():
                speech = row["speech"]
                f.write(f"{idx}\t{speech}\n")
        
        print("Initializing OpenAI client ...")
        client = openai.OpenAI()
        
        print("Extracting subtopics and phrases using LLM ...")
        all_subtopics = []
        all_phrase_lists = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            speech = row["speech"]
            policy = str(row["policy_area"])  # convert to string
            subtopic, phrase_list = extract_topic_and_phrases(speech, policy, client)
            all_subtopics.append(subtopic)
            all_phrase_lists.append(phrase_list)
        
        # Create training data rows: one row per key phrase.
        rows = []
        for idx, row in df.iterrows():
            speech = row["speech"]
            policy = str(row["policy_area"])
            subtopic = all_subtopics[idx]
            phrases = all_phrase_lists[idx]
            for phrase in phrases:
                rows.append({
                    "document": speech,
                    "policy_area": policy,
                    "subtopic": subtopic,
                    "phrase": phrase
                })
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)
        print(f"Preprocessing complete. Training data saved to {output_path}.")
    
    # Build a global mapping of parent topics from the "policy_area" column.
    # Convert each value to string and filter out nulls.
    parent_topics = sorted([str(x) for x in pd.read_csv(args.input_file)["policy_area"].unique().tolist() if pd.notnull(x)])
    topic_to_topic_idx = {topic: idx for idx, topic in enumerate(parent_topics)}
    topics_out = os.path.join(args.data_dir, "topics.txt")
    with open(topics_out, "w", encoding="utf-8") as f:
        for topic, idx in topic_to_topic_idx.items():
            f.write(f"{idx}\t{topic}\n")
    print(f"Parent topics mapping saved to {topics_out}")

if __name__ == "__main__":
    main()