#!/usr/bin/env python
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
    For a given speech and its broad policy area, call the LLM to extract a short, general topic (1–5 words) describing the main issue in the speech
    Retries up to max_attempts until successful.
    Returns topic
    """
    prompt = (
        "Here we have the text of a congressional speech and a broad policy area assigned by the Congressional Research Service. For the following speech, please output only a short topic (1-5 words) that describes the main political issue discussed in the speech. The topic should be inherently stanced as something one could take a for or against position on. For example, you should output Budget Cuts instead of Budget Policy.\n\n"
        f"Speech: {speech}\n\n"
        f"Broad Policy Area: {policy}\n\n"
        "Response:\n"
    )
    messages = [
        {"role": "system",
         "content": (
             "You are a helpful assistant that extracts the main topic from congressional speeches. "
             "For the following speech, please output only a topic (1-5 words) that describes the main political issue discussed in the speech. The topic should be inherently stanced as something one could take a for or against position on, such as Gun Control instead of Gun Controversy or Gun Policy. For example, you should output Budget Cuts instead of Budget Policy. Make it a single topic (for example, Welfare Programs instead of Social Security and Medicare)."
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
    parser = argparse.ArgumentParser(description="Preprocess df_bills.csv to produce training_data.csv")
    parser.add_argument('--input-file', type=str, default='df_bills.csv',
                        help="Input CSV file")
    parser.add_argument('--output-file', type=str, default='training_data.csv',
                        help="Output CSV file")
    parser.add_argument('--data-dir', type=str, default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao',
                        help="Directory of files")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    input_path = os.path.join(args.data_dir, args.input_file)
    output_path = os.path.join(args.data_dir, args.output_file)
    cache_path = os.path.join(args.data_dir, f"cache_{args.input_file}.csv")
    
    print("Initializing OpenAI client ...")
    client = openai.OpenAI(api_key=api_keys.OPENAI_API_KEY)
    
    print("Extracting topics using LLM ...")
    all_topics = []
    df = pd.read_csv(input_path)
    # drop rows where policy_area is na
    original_df_length = len(df)
    df = df.dropna(subset=["policy_area"])
    print(f"Original df length: {original_df_length}")
    print(f"New df length: {len(df)}")
    print(f"Percentage dropped: {(original_df_length - len(df)) / original_df_length * 100:.2f}%")
    # print all policy areas
    print(df["policy_area"].unique())

    rows = []

    if not os.path.exists(cache_path):
        for i, row in tqdm(df.iterrows(), total=len(df)):
            speech = row["speech"]
            policy = str(row["policy_area"])
            topic = extract_topic(speech, policy, client)
            all_topics.append(topic)

        if len(all_topics) != len(df):
            print(f"Error: Number of topics ({len(all_topics)}) does not match number of rows in df ({len(df)})")

        # write all topics to csv, with each topic on a row
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["topic"])
            writer.writerows([[topic] for topic in all_topics])
    else:
        # read in all_topics
        all_topics = pd.read_csv(cache_path)["topic"].tolist()

    for (idx, row), topic in zip(df.iterrows(), all_topics):
        speech = row["speech"]
        speaker = row["speaker"]
        chamber = row["chamber"]
        policy = str(row["policy_area"])
        summary = row["latest_summary"]
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

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Training data saved to {output_path}.")

if __name__ == "__main__":
    main()