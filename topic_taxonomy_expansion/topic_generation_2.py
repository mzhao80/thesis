#!/usr/bin/env python
"""
Topic Generation - Second Level

This script generates second-level subtopics for each document based on the clustered first-level subtopics.
It builds upon the results of clustering.py (step_2.csv) to create a more detailed taxonomy level.

The script:
1. Loads the clustering results from step_2.csv and original document data from step_1.csv
2. For each document, generates a more specific subtopic within its assigned first-level topic cluster
3. Handles missing or empty predictions through regeneration
4. Outputs the results to a CSV file for subsequent clustering

Input:
- step_1.csv: Original data with first-level subtopics
- step_2.csv: Clustering results from first level

Output:
- step_3.csv: Contains policy areas, first-level subtopics, and second-level subtopics
"""

import os
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm.auto import tqdm

def main():
    """
    Main function to generate second-level subtopics for the taxonomy.
    """
    # Define paths
    taxonomy_file = "step_2.csv"   # Taxonomy with gpt_label and source_indices
    original_file = "step_1.csv"   # Original CSV containing the "document" column
    model_dir = "./lora_bart_topics_2023"  # Directory where the trained BART model was saved
    
    # Load the taxonomy and original CSV
    print("Loading taxonomy and original data...")
    taxonomy_df = pd.read_csv(taxonomy_file)
    original_df = pd.read_csv(original_file)
    
    # Load the fine-tuned BART model and its tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Prepare the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")
    
    results = []
    count = 0
    max_retries = 5
    
    # Iterate over each taxonomy row
    print("Processing taxonomy entries...")
    for _, row in tqdm(taxonomy_df.iterrows(), desc="Processing taxonomy", total=len(taxonomy_df)):
        gpt_label = row["gpt_label"]  # First-level subtopic (parent topic for second level)
        source_indices_str = row["source_indices"]  # Semicolon-separated indices
        
        # Parse source indices into a list of integers
        indices = [int(i) for i in source_indices_str.split(";")]
        
        # For each corresponding document, process individually
        for doc_idx in indices:
            count += 1
            # Handle "Misc." topics directly without model inference
            if gpt_label == "Misc.":
                results.append({
                    "matching_idx": doc_idx,
                    "policy_area": row["policy_area"],
                    "subtopic_1": gpt_label,
                    "subtopic_2": "Misc."
                })
                continue

            # Extract document text for processing
            doc_text = original_df.loc[doc_idx, "document"]
            prompt = f"Extract a detailed subtopic of the parent topic, {gpt_label}, that represents the main policy issue in the following text: {doc_text}"
            
            # Tokenize and move to device
            encoded = tokenizer(prompt, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Generate prediction with retries for empty results
            retries = 0
            pred = ""
            while pred.strip() == "" and retries < max_retries:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        min_length=3,
                        max_length=16,
                        length_penalty=4.0,
                        early_stopping=True
                    )
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                retries += 1

            # If failed after max retries, use "Misc."
            if retries == max_retries and pred.strip() == "":
                pred = "Misc."
            
            # Store result
            results.append({
                "matching_idx": doc_idx,
                "policy_area": row["policy_area"],
                "subtopic_1": gpt_label,
                "subtopic_2": pred
            })
    
    # Save the results to CSV
    print("Saving results...")
    new_df = pd.DataFrame(results)
    # Sort by document index
    new_df = new_df.sort_values(by="matching_idx")
    output_csv = "step_3.csv"
    new_df.to_csv(output_csv, index=False)
    print(f"Processed {count} documents.")
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    main()
