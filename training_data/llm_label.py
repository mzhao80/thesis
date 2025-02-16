#!/usr/bin/env python
import os
import argparse
import pandas as pd
from tqdm import tqdm
import openai
import api_keys

def extract_topic(speech, policy, client, max_attempts=10):
    """
    For a given speech and its broad policy area, call the LLM to extract a short, general topic (1–5 words) describing the main issue in the speech
    Retries up to max_attempts until successful.
    Returns (topic, phrase)
    """
    prompt = (
        "Here we have the text of a congressional speech and a broad policy area assigned by the Congressional Research Service. For the following speech, please output only a short, general topic (1–5 words) that describes the main political issue discussed in the speech. It should be general and unstanced, for example Budget Cuts instead of Opposition to Budget Cuts.\n"
        f"Speech: {speech}\n\n"
        f"Broad Policy Area: {policy}\n\n"
        "Response:\n"
    )
    messages = [
        {"role": "system",
         "content": (
             "You are a helpful assistant that extracts the main topic from congressional speeches. "
             "For the following speech, please output only a short, general topic (1–5 words) that describes the main political issue discussed in the speech. It should be general and unstanced, for example Budget Cuts instead of Opposition to Budget Cuts."
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
            subtopic = response.choices[0].message.content.strip()
            return subtopic, phrase
        except Exception as e:
            print(f"Error parsing LLM response on attempt {attempt+1}: {e}\nResponse was: {content if 'content' in locals() else 'N/A'}")
            attempt += 1
    return ""

def main():
    parser = argparse.ArgumentParser(description="Preprocess df_bills.csv to produce training_data.csv")
    parser.add_argument('--input-file', type=str, default='df_bills.csv',
                        help="Input CSV file (must contain 'speech' and 'policy_area' columns)")
    parser.add_argument('--output-file', type=str, default='training_data.csv',
                        help="Output CSV file with document, policy_area, topic, and phrase columns")
    parser.add_argument('--data-dir', type=str, default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao',
                        help="Directory of files")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    input_path = os.path.join(args.data_dir, args.input_file)
    output_path = os.path.join(args.data_dir, args.output_file)
    
    print("Initializing OpenAI client ...")
    client = openai.OpenAI(api_key=api_keys.OPENAI_API_KEY)
    
    print("Extracting topics using LLM ...")
    all_topics = []
    df = pd.read_csv(input_path)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        speech = row["speech"]
        policy = str(row["policy_area"])
        topic = extract_topic(speech, policy, client)
        all_topics.append(topic)
    
    # Create training data rows: one row per key phrase.
    rows = []
    for idx, row in df.iterrows():
        speech = row["speech"]
        policy = str(row["policy_area"])
        topic = all_topics[idx]   
        rows.append({
                "document": speech,
                "policy_area": policy,
                "topic": topic
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Training data saved to {output_path}.")

if __name__ == "__main__":
    main()