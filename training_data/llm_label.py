#!/usr/bin/env python
"""
Congressional Speech Topic Labeling Module

This script uses the OpenAI GPT API to automatically extract and label the main political
topic from congressional speeches. It takes a CSV file containing speech data and policy areas
as input and outputs a CSV file with the labeled topics.

Usage:
    python llm_label.py [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--data-dir DATA_DIR]

Arguments:
    --input-file: Path to input CSV file (default: df_bills.csv)
    --output-file: Path to output CSV file (default: training_data.csv)
    --data-dir: Directory containing data files (default: /n/holylabs/LABS/arielpro_lab/Lab/michaelzhao)
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import openai
import api_keys
from bs4 import BeautifulSoup
import csv

def extract_topic(speech, policy, client, max_attempts=10):
    """
    Extract a short, general topic describing the main issue in a congressional speech.
    
    Uses GPT to generate a concise (2-5 word) topic that describes the main political issue
    in the speech. The topic is designed to be inherently stanced (something one could take
    a position on). The function will retry up to max_attempts if the API call fails.
    
    Args:
        speech (str): The text of the congressional speech
        policy (str): The broad policy area assigned by the Congressional Research Service
        client (openai.OpenAI): Initialized OpenAI client
        max_attempts (int): Maximum number of attempts to call the API
        
    Returns:
        str: The extracted topic, or an empty string if all attempts fail
    """
    prompt = (
        "Here we have the text of a congressional speech and a broad policy area assigned by the Congressional Research Service. For the following speech, please output only a short topic (2-5 words) that describes the main political issue discussed in the speech. The topic should be inherently stanced as something one could take a for or against position on. For example, you should output Budget Cuts instead of Budget Policy.\n\n"
        f"Speech: {speech}\n\n"
        f"Broad Policy Area: {policy}\n\n"
        "Response:\n"
    )
    messages = [
        {"role": "system",
         "content": (
             "You are a helpful assistant that extracts the main topic from congressional speeches. "
             "For the following speech, please output only a topic (2-5 words) that describes the main political issue discussed in the speech. The topic should be inherently stanced as something one could take a for or against position on, such as Gun Control instead of Gun Controversy or Gun Policy. For example, you should output Budget Cuts instead of Budget Policy. Make it a single topic (for example, Welfare Programs instead of Social Security and Medicare)."
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
            return subtopic
        except Exception as e:
            print(f"Error parsing LLM response on attempt {attempt+1}: {e}\nResponse was: {content if 'content' in locals() else 'N/A'}")
            attempt += 1
    return ""

def main():
    """
    Main function to process speech data and extract topics.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract topics from congressional speeches using GPT")
    parser.add_argument('--input-file', type=str, default='df_bills.csv',
                        help="Input CSV file containing speech data")
    parser.add_argument('--output-file', type=str, default='training_data.csv',
                        help="Output CSV file to save labeled data")
    parser.add_argument('--data-dir', type=str, default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao',
                        help="Directory containing data files")
    args = parser.parse_args()
    
    # Set up paths
    os.makedirs(args.data_dir, exist_ok=True)
    input_path = os.path.join(args.data_dir, args.input_file)
    output_path = os.path.join(args.data_dir, args.output_file)
    cache_path = os.path.join(args.data_dir, f"cache_{args.input_file}.csv")
    
    # Initialize OpenAI client
    print("Initializing OpenAI client ...")
    client = openai.OpenAI(api_key=api_keys.OPENAI_API_KEY)
    
    # Load and preprocess data
    print("Loading data and extracting topics using LLM ...")
    df = pd.read_csv(input_path)
    
    # Drop rows where policy_area is None
    original_df_length = len(df)
    df = df.dropna(subset=["policy_area"])
    print(f"Original dataset length: {original_df_length}")
    print(f"Dataset length after removing rows with missing policy areas: {len(df)}")
    print(f"Percentage of rows dropped: {(original_df_length - len(df)) / original_df_length * 100:.2f}%")
    
    # Display unique policy areas for reference
    print("Unique policy areas in the dataset:")
    print(df["policy_area"].unique())

    all_topics = []
    rows = []

    # Check if cached topics exist, otherwise generate new ones
    if not os.path.exists(cache_path):
        print("No cached topics found. Generating new topics...")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            speech = row["speech"]
            policy = str(row["policy_area"])
            topic = extract_topic(speech, policy, client)
            all_topics.append(topic)

        if len(all_topics) != len(df):
            print(f"Error: Number of topics ({len(all_topics)}) does not match number of rows in df ({len(df)})")

        # Save topics to cache file
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["topic"])
            writer.writerows([[topic] for topic in all_topics])
    else:
        print(f"Loading cached topics from {cache_path}")
        all_topics = pd.read_csv(cache_path)["topic"].tolist()

    # Combine original data with generated topics
    for (idx, row), topic in zip(df.iterrows(), all_topics):
        speech = row["speech"]
        speaker = row["speaker"]
        chamber = row["chamber"]
        policy = str(row["policy_area"])
        summary = row["latest_summary"]
        
        # Clean HTML from summary if present
        clean_summary = "" if pd.isna(summary) else BeautifulSoup(summary, "html.parser").get_text(separator=" ").strip()
        leg_subjects = row["legislative_subjects"]
        committees = row["committees"]
        
        rows.append({
            "document": speech,
            "speaker": speaker,
            "chamber": chamber,
            "policy_area": policy,
            "topic": topic,
            "legislative_subjects": leg_subjects,
            "committees": committees,
            "summary": clean_summary,
        })

    # Save processed data to output file
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Topic labeling complete. Data saved to {output_path}.")

if __name__ == "__main__":
    main()